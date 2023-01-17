! pip install pyspark pyhash networkx

! ls /kaggle/input/



from pyspark.sql import SparkSession, functions as F

import matplotlib.pyplot as plt



spark = (

    SparkSession.builder

    .config("spark.driver.memory", "12g")

    .getOrCreate()

)

spark.conf.set("spark.sql.shuffle.partitions", spark.sparkContext.defaultParallelism*2)

spark.conf.get("spark.driver.memory")
input_path = "/kaggle/input/cord19-parquet/cord19.parquet"

df = spark.read.parquet(input_path)

df.createOrReplaceTempView("cord19")

df.printSchema()



input_path = "/kaggle/input/cord19-parquet/metadata.parquet"

df = spark.read.parquet(input_path)

df.createOrReplaceTempView("metadata")

df.printSchema()

df.show(vertical=True, n=1)
query = """

select

    paper_id,

    metadata.title as paper,

    value.title as citation

from

    cord19

lateral view

    explode(bib_entries) as key, value

"""

references = spark.sql(query)



print("writing references to parquet")

# A single file reduces parallelism, but is more efficient for output

%time references.toPandas().to_parquet("references.parquet")

references = spark.read.parquet("references.parquet")

references.createOrReplaceTempView("references")

references.show(n=5)

print(f"references has {references.count()} rows")
query = """

select

    citation,

    count(distinct paper) as num_citations

from

    references

group by

    citation

order by

    num_citations desc

"""

spark.sql(query).show(n=10, truncate=60)



spark.sql(query).groupBy("num_citations").count().toPandas().plot.scatter("num_citations", "count")

plt.xscale("log")

plt.yscale("log")

plt.title("degree distribution of citation network w/o deduplication")

plt.show()
query = """

with titles as (

    select paper as title from references

    union

    select citation as title from references

)

select

    title,

    sha1(title) as title_sha,

    count(*) as num_cited

from

    titles

where

    length(title) > 0

group by

    title,

    title_sha

"""

titles = spark.sql(query)



print("writing to titles to parquet")

%time titles.toPandas().to_parquet("titles.parquet")



titles = spark.read.parquet("titles.parquet")

print(f"there are {titles.count()} distinct titles")

titles.show(n=5, truncate=60)



(

    titles

    .withColumn("length", F.expr("log2(length(title))"))

    .groupBy("length")

    .count()

    .select("length")

).toPandas().hist("length", density=1)

plt.title("histogram of log_2(length)")

plt.show()
import pyhash

import sys



hasher = pyhash.murmur3_32()

value = "Control of Communicable Diseases Manual"

hashed = hasher(value)

print(f"hashed '{value}' into {sys.getsizeof(hashed)} bits: {hashed}")
from pyspark.ml.linalg import Vectors, VectorUDT

import pyhash





@F.udf(VectorUDT())

def hashed_shingles(text, k=9):

    """"Generates a set of hashed k-shingles as a sparse matrix.

    

    Text is lower cased before it is shingled. Punctuation is is

    assumed to be significant.



    The max input dimension is log2(2147483647), or 30.99. This

    determines the number of buckets. To calculate the empirical

    limits of a SparseVector, set `num_buckets` to 2**32-1. Then 

    refer to the resulting exception message.

    """

    num_buckets = 2**24

    

    # A standard library alternative is to use 

    # a check summing routine. This will be slower.

    # import zlib; hasher = zlib.adler32

    hasher = pyhash.murmur3_32()

    

    shingles = (text[i:i+k].lower() for i in range(len(text)-k+1))

    hashed_shingles = {(hasher(s) % num_buckets, 1) for s in shingles}

    

    return Vectors.sparse(num_buckets, hashed_shingles)
from pyspark.ml.feature import MinHashLSH



num_shingles=9

num_hash_tables=5

minhasher = MinHashLSH(

    inputCol="features", 

    outputCol="hashes", 

    numHashTables=num_hash_tables

)



prepared_titles = (

    titles

    .withColumn("length", F.expr("length(title)"))

    .where(f"length(title) >= {num_shingles}")

    .withColumn("features", hashed_shingles("title"))

)



model = minhasher.fit(prepared_titles)

hashed_titles = model.transform(prepared_titles)
threshold = 0.8

hashed_titles.cache()



ann_titles = (

    model.approxSimilarityJoin(

        hashed_titles, 

        hashed_titles, 

        threshold,

        distCol="jaccard_distance"

    )

    .where("datasetA.title_sha <> datasetB.title_sha")

    .orderBy("jaccard_distance")

)
edgelist = (

    ann_titles

    .select(

        F.col("datasetA.title_sha").alias("src"), 

        F.col("datasetB.title_sha").alias("dst"), 

        "jaccard_distance"

    )

)

edgelist.cache()



%time print(f"there are {edgelist.count()} edges")
import networkx as nx

import pandas as pd

from tqdm import tqdm





cc = []

for i in tqdm(range(0, 85, 10)):

    graph = nx.from_pandas_edgelist(

        edgelist.where(f"jaccard_distance <= {i/100}").toPandas(),

        source="src",

        target="dst",

        edge_attr="jaccard_distance",

    )

    cc.append(

        dict(

            threshold=i/100,

            nodes=len(graph.nodes),

            edges=len(graph.edges),

            connected_components=nx.number_connected_components(graph),

            max_connected_component=max(map(len, nx.connected_components(graph))),

            average_clustering=nx.average_clustering(graph),

        )

    )

df = pd.DataFrame(cc)

df
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

df.plot("threshold", "nodes", ax=axes[0][0])

df.plot("threshold", "edges", ax=axes[0][1])

df.plot("threshold", "connected_components", ax=axes[1][0])

df.plot("threshold", "max_connected_component", ax=axes[1][1])

fig.suptitle("effect of threshold on network construction")

plt.show()
from pyspark.sql import Window

from pyspark.sql import types as T





def index_with_dedupe_threshold(

    threshold, edgelist, title_sha, output_path, n=10

):

    output_prefix = f"{output_path}/t{round(threshold*100):02d}"

    graph = nx.from_pandas_edgelist(

        edgelist.where(f"jaccard_distance <= {threshold}").toPandas(),

        source="src",

        target="dst",

        edge_attr="jaccard_distance",

    )

    

    # index of approximate citation to the canonical citation

    title_index = spark.sparkContext.parallelize(

        [(max(nodes), list(nodes)) for nodes in nx.connected_components(graph)]

    ).toDF(

        schema=T.StructType(

            [

                T.StructField("citation_id", T.StringType()),

                T.StructField("approx_citation_ids", T.ArrayType(T.StringType())),

            ]

        )

    )

    citation_index = (

        title_index.withColumn("near_duplicates", F.size("approx_citation_ids"))

        .withColumn("rank", F.row_number().over(Window.orderBy(F.desc("near_duplicates"))))

    )

    citation_index.cache()

    output = f"{output_prefix}_citation_index.parquet"

    citation_index.toPandas().to_parquet(output)



    # map citation_ids to titles

    indexed_titles = (

        citation_index

        .withColumn("approx_citation_id", F.explode("approx_citation_ids"))

        .drop("approx_citation_ids")

        .join(

            title_sha.selectExpr("title as approx_citation", "title_sha as approx_citation_id"),

            on="approx_citation_id",

        )

        .join(

            title_sha.selectExpr(

                "title as citation", "title_sha as citation_id"

            ),

            on="citation_id",

        )

        .withColumn(

            "edit_distance", F.levenshtein("citation", "approx_citation")

        )

        .orderBy("rank", F.desc("edit_distance"))

    )

    indexed_titles.cache()

    

    # write out the titles and their edit distances

    output = f"{output_prefix}_approx_citation.csv"

    (

        indexed_titles

        .withColumn("secondary_rank", F.row_number().over(

                Window.partitionBy("rank").orderBy(F.desc("near_duplicates"))

            )

        )

        # only keep the top 200 results for each cluster

        .where("secondary_rank < 200")

        .where("rank <= 50")

        .select("rank", "edit_distance", "approx_citation")

        .orderBy("rank", F.desc("edit_distance"))

        .toPandas()

        .to_csv(output)

    )

    

    # write out all of the titles by rank

    output = f"{output_prefix}_ranked_titles.csv"

    (

        indexed_titles

        .groupBy("rank")

        .agg(F.count("*").alias("near_duplications"), F.max("citation").alias("citation"))

        .orderBy("rank")

        .toPandas()

        .to_csv(output)

    )

    

    # write out stats for each cluster

    output = f"{output_prefix}_edit_distance_stats.csv"

    stats = (

        indexed_titles

        .withColumn("x", F.col("edit_distance"))

        .groupBy("rank")

        .agg(

            F.count("x").alias("count"),

            F.mean("x").alias("mean"),

            F.stddev("x").alias("stddev"),

            F.min("x").alias("min"),

            F.expr("percentile(x, array(0.5))[0] as p50"),

            F.expr("percentile(x, array(0.9))[0] as p90"),

            F.max("x").alias("max"),

        )

        .orderBy("rank")

    )

    rounded = stats.select([F.round(c, 2).alias(c) for c in stats.columns])

    rounded.toPandas().to_csv(output)

    

    citation_index.unpersist()

    indexed_titles.unpersist()
# ensure output folder exists

! mkdir -p corrections



for i in tqdm(range(0, 85, 10)):

    index_with_dedupe_threshold(i/100, edgelist, titles, "corrections", 10)
df = (

    spark.read.csv("corrections/*_edit_distance_stats.csv", header=True)

    .withColumn("threshold",

        F.regexp_extract(F.input_file_name(), ".*t(\d+)_edit_distance_stats.csv", 1)

        .astype("float") / 100

    )

)

df.show(n=5,truncate=False)

output = "edit_distance_stats.csv"

print(f"writing out {output}")

df.drop("_c0").toPandas().to_csv(output)



(

    df.groupBy("threshold")

    .agg(F.expr("mean(cast(p50 as float)) as mean_p50"))

    .orderBy("threshold")

).toPandas().plot("threshold", "mean_p50")

plt.show()
for threshold in [0.2, 0.5, 0.8]:

    fig, axes = plt.subplots(1, 2, figsize=(10, 3))

    fig.suptitle(f"threshold={threshold}")

    (

        df

        .where(f"threshold={threshold}")

        .selectExpr("log2(cast(p50 as float)) as median_edit_distance")

        .toPandas().hist("median_edit_distance", bins=10, ax=axes[0], density=1.0)

    )

    (

        df

        .where(f"threshold={threshold}")

        .selectExpr("cast(count as float) as degree")

        .groupBy("degree")

        .count()

        .toPandas().plot.scatter("degree", "count", ax=axes[1], title="degree distribution")

    )

    plt.xscale("log")

    plt.yscale("log")

    plt.show()
! cp corrections/t30_citation_index.parquet citation_index.parquet

citation_index = spark.read.parquet("citation_index.parquet")



print(f"references:\t{references}")

print(f"citation_index:\t{citation_index}")
query = """

with papers as (

    select distinct

        paper_id,

        paper

    from

        references

)

select 

    paper_id = sha1(paper) sha_matches_paper_id,

    count(*) as num_seen

from

    papers

group by

    sha_matches_paper_id

"""

spark.sql(query).show(truncate=False)



query = """

select distinct

    paper_id,

    sha1(paper)

from

    references

limit 5

"""

spark.sql(query).show(truncate=False)

index = citation_index.select("citation_id", F.explode("approx_citation_ids").alias("approx_citation_id"))

citations = (

    references

    .withColumn("paper_sha", F.sha1("paper"))

    .withColumn("citation_sha", F.sha1("citation"))

    # left join, since the citation index only contains duplicates. If the title is unique, we'll use

    # the title hash as the id for the node.

    .join(

        index.selectExpr("citation_id as src_id", "approx_citation_id as paper_sha"),

        on="paper_sha",

        how="left",

    )

    .withColumn("src_citation_id", F.coalesce("src_id", "paper_sha"))

    .join(

        index.selectExpr("citation_id as dst_id", "approx_citation_id as citation_sha"),

        on="citation_sha",

        how="left",

    )

    .withColumn("dst_citation_id", F.coalesce("dst_id", "citation_sha"))

    .drop("paper_sha", "citation_sha")

    .select("paper_id", "src_citation_id", "dst_citation_id", "paper", "citation")

    # remove self-edges and duplicate edges

    .where("src_citation_id <> dst_citation_id")

    .distinct()

    .orderBy("paper_id", "src_citation_id")

)



print("writing out citations.parquet")

%time citations.toPandas().to_parquet("citations.parquet")



citations.limit(5).toPandas()
# We're done with Spark, lets try to free up memory

spark.stop()



# kill java to make memory immediately available

! killall -9 java



# Also free up any dangling objects

import gc

gc.collect() 
import pandas as pd

import networkx as nx



citations = pd.read_parquet("citations.parquet")

citations.head()
pd.set_option('display.max_colwidth', 120)



G = nx.from_pandas_edgelist(citations, source="paper", target="citation")

print(nx.info(G))

%time pr = nx.pagerank(G)

(

    pd.DataFrame(pr.items(), columns=["citation_id", "pagerank"])

    .sort_values(by="pagerank", ascending=False)

    .head(20)

)
pd.set_option('display.max_colwidth', 120)



G = nx.from_pandas_edgelist(citations, source="src_citation_id", target="dst_citation_id")

print(nx.info(G))

%time pr = nx.pagerank(G)

pr_df = pd.DataFrame(pr.items(), columns=["citation_id", "pagerank"]).set_index("citation_id")

pr_df.to_csv("citation_pagerank.csv")

(

    citations[["dst_citation_id", "citation"]]

    .rename(columns={"dst_citation_id": "citation_id"})

    .groupby("citation_id")

    .first()

    .join(pr_df, on="citation_id", how="inner")

    .sort_values(by="pagerank", ascending=False)

    .reset_index()

    .head(20)[["citation", "pagerank"]]

)