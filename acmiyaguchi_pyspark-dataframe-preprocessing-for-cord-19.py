! pip install pyspark
from pyspark.sql.functions import lit

from pyspark.sql.types import (

    ArrayType,

    IntegerType,

    MapType,

    StringType,

    StructField,

    StructType,

)





def generate_cord19_schema():

    """Generate a Spark schema based on the semi-textual description of CORD-19 Dataset.



    This captures most of the structure from the crawled documents, and has been

    tested with the 2020-03-13 dump provided by the CORD-19 Kaggle competition.

    The schema is available at [1], and is also provided in a copy of the

    challenge dataset.



    One improvement that could be made to the original schema is to write it as

    JSON schema, which could be used to validate the structure of the dumps. I

    also noticed that the schema incorrectly nests fields that appear after the

    `metadata` section e.g. `abstract`.

    

    [1] https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-03-13/json_schema.txt

    """



    # shared by `metadata.authors` and `bib_entries.[].authors`

    author_fields = [

        StructField("first", StringType()),

        StructField("middle", ArrayType(StringType())),

        StructField("last", StringType()),

        StructField("suffix", StringType()),

    ]



    authors_schema = ArrayType(

        StructType(

            author_fields

            + [

                # Uncomment to cast field into a JSON string. This field is not

                # well-specified in the source.

                StructField(

                    "affiliation",

                    StructType(

                        [

                            StructField("laboratory", StringType()),

                            StructField("institution", StringType()),

                            StructField(

                                "location",

                                StructType(

                                    [

                                        StructField("settlement", StringType()),

                                        StructField("country", StringType()),

                                    ]

                                ),

                            ),

                        ]

                    ),

                ),

                StructField("email", StringType()),

            ]

        )

    )



    # used in `section_schema` for citations, references, and equations

    spans_schema = ArrayType(

        StructType(

            [

                # character indices of inline citations

                StructField("start", IntegerType()),

                StructField("end", IntegerType()),

                StructField("text", StringType()),

                StructField("ref_id", StringType()),

            ]

        )

    )



    # A section of the paper, which includes the abstract, body, and back matter.

    section_schema = ArrayType(

        StructType(

            [

                StructField("text", StringType()),

                StructField("cite_spans", spans_schema),

                StructField("ref_spans", spans_schema),

                # While equations don't appear in the abstract, but appear here

                # for consistency

                StructField("eq_spans", spans_schema),

                StructField("section", StringType()),

            ]

        )

    )



    bib_schema = MapType(

        StringType(),

        StructType(

            [

                StructField("ref_id", StringType()),

                StructField("title", StringType()),

                StructField("authors", ArrayType(StructType(author_fields))),

                StructField("year", IntegerType()),

                StructField("venue", StringType()),

                StructField("volume", StringType()),

                StructField("issn", StringType()),

                StructField("pages", StringType()),

                StructField(

                    "other_ids",

                    StructType([StructField("DOI", ArrayType(StringType()))]),

                ),

            ]

        ),

        True,

    )



    # Can be one of table or figure captions

    ref_schema = MapType(

        StringType(),

        StructType(

            [

                StructField("text", StringType()),

                # Likely equation spans, not included in source schema, but

                # appears in JSON

                StructField("latex", StringType()),

                StructField("type", StringType()),

            ]

        ),

    )



    return StructType(

        [

            StructField("paper_id", StringType()),

            StructField(

                "metadata",

                StructType(

                    [

                        StructField("title", StringType()),

                        StructField("authors", authors_schema),

                    ]

                ),

                True,

            ),

            StructField("abstract", section_schema),

            StructField("body_text", section_schema),

            StructField("bib_entries", bib_schema),

            StructField("ref_entries", ref_schema),

            StructField("back_matter", section_schema),

        ]

    )





def extract_dataframe_kaggle(spark):

    """Extract a structured DataFrame from the semi-structured document dump.



    It should be fairly straightforward to modify this once there are new

    documents available. The date of availability (`crawl_date`) and `source`

    are available as metadata.

    """

    base = "/kaggle/input/CORD-19-research-challenge"

    crawled_date = "2020-03-13"

    sources = [

        "noncomm_use_subset",

        "comm_use_subset",

        "biorxiv_medrxiv",

        "pmc_custom_license",

    ]



    dataframe = None

    for source in sources:

        path = f"{base}/{crawled_date}/{source}/{source}"

        df = (

            spark.read.json(path, schema=generate_cord19_schema(), multiLine=True)

            .withColumn("crawled_date", lit(crawled_date))

            .withColumn("source", lit(source))

        )

        if not dataframe:

            dataframe = df

        else:

            dataframe = dataframe.union(df)

    return dataframe

from pyspark.sql import SparkSession

from pyspark.sql import functions as F



spark = SparkSession.builder.getOrCreate()

df = extract_dataframe_kaggle(spark)

df.printSchema()



df.createOrReplaceTempView("cord19")
print("Using the Spark DataFrame interface...")

df.groupBy("source").agg(F.countDistinct("paper_id")).show()



print("Using the Spark SQL interface...")

query = """

SELECT

    source,

    COUNT(DISTINCT paper_id)

FROM

    cord19

GROUP BY

    source

"""

spark.sql(query).show()
authors = df.select("paper_id", F.explode("metadata.authors").alias("author")).select("paper_id", "author.*")

authors.select("first", "middle", "last", "email").where("email <> ''").show(n=5)

authors.printSchema()
(

    authors.groupBy("first", "middle", "last")

    .agg(F.countDistinct("paper_id").alias("n_papers"))

    .orderBy(F.desc("n_papers"))

).show(n=5)
query = """

WITH authors AS (

    SELECT

        paper_id,

        author.*

    FROM

        cord19

    LATERAL VIEW

        explode(metadata.authors) AS author

)

SELECT

    first,

    last,

    COUNT(DISTINCT paper_id) as n_papers

FROM

    authors

GROUP BY

    first,

    last

ORDER BY

    n_papers DESC

"""



spark.sql(query).show(n=5)
# based on https://stackoverflow.com/a/50668635

from pyspark.sql import Window



abstract = (

    df.select("paper_id", F.posexplode("abstract").alias("pos", "value"))

    .select("paper_id", "pos", "value.text")

    .withColumn("ordered_text", F.collect_list("text").over(Window.partitionBy("paper_id").orderBy("pos")))

    .groupBy("paper_id")

    .agg(F.max("ordered_text").alias("sentences"))

    .select("paper_id", F.array_join("sentences", " ").alias("abstract"))

    .withColumn("words", F.size(F.split("abstract", "\s+")))

)



abstract.show(n=5)
abstract.explain()
query = """

WITH abstract AS (

    SELECT

        paper_id,

        pos,

        value.text as text

    FROM

        cord19

    LATERAL VIEW

        posexplode(abstract) AS pos, value

),

collected AS (

    SELECT

        paper_id,

        collect_list(text) OVER (PARTITION BY paper_id ORDER BY pos) as sentences

    FROM

        abstract

),

sentences AS (

    SELECT

        paper_id,

        max(sentences) as sentences

    FROM

        collected

    GROUP BY

        paper_id

)

SELECT

    paper_id,

    array_join(sentences, " ") as abstract,

    -- make sure the regex is being escaped properly

    size(split(array_join(sentences, " "), "\\\s+")) as words

FROM

    sentences

"""



spark.sql(query).show(n=5)
@F.udf("string")

def join_abstract(rows) -> str:

    return " ".join([row.text for row in rows])



(

    df.select("paper_id", join_abstract("abstract").alias("abstract"))

    .where("abstract <> ''")

    # mix and match SQL using `pyspark.sql.functions.expr` or `DataFrame.selectExpr`

    .withColumn("words", F.expr("size(split(abstract, '\\\s+'))"))

).show(n=5)
spark.udf.register("join_abstract", join_abstract)



query = """

SELECT

    paper_id,

    join_abstract(abstract) as abstract,

    size(split(join_abstract(abstract), '\\\s+')) as words

FROM

    cord19

WHERE

    size(abstract) > 1

"""



spark.sql(query).show(n=5)