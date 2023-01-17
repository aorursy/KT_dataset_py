! pip install pyspark[sql] git+https://github.com/lmcinnes/umap.git@0.4.0rc2
from pyspark.sql import SparkSession, functions as F, types as T

from pyspark.ml.linalg import Vectors, VectorUDT

import pandas as pd

import matplotlib.pyplot as plt

import networkx as nx

import numpy as np

import scipy.sparse as sp





def create_graph(citations):

    return nx.from_pandas_edgelist(

        citations.select("src_citation_id", "dst_citation_id").distinct().toPandas(), 

        source="src_citation_id",

        target="dst_citation_id",

        create_using=nx.DiGraph

    )





def laplacian_embedding(g_nx, dim):

    """Compute the smallest-but-1 eigenvectors of the laplacian normalized by the variance.

    

    https://www2.imm.dtu.dk/projects/manifold/Papers/Laplacian.pdf

    """

    L = nx.laplacian_matrix(g_nx).astype("float64")

    # returns ordered by the smallest eigenvalues

    w, v = sp.linalg.eigsh(L, k=dim + 1)

    return np.divide(v[:, 1:], np.sqrt(w[1:]))[::-1]





def create_embedding(dataframe, dim=8):

    G = create_graph(dataframe)

    print(nx.info(G))

    

    emb = laplacian_embedding(G.to_undirected(), dim)

    print(f"{emb.shape} -> {emb.shape[0]*emb.shape[1]*8/10**6} mb")

    

    rows = zip(G.nodes(), map(Vectors.dense, emb))

    return spark.createDataFrame(rows, schema=["citation_id", "vector"])





@F.udf(VectorUDT())

def udf_average_embedding(neighbors):

    # What is the overhead? Could this be done in a Pandas UDF?

    # https://databricks.com/blog/2017/10/30/introducing-vectorized-udfs-for-pyspark.html

    # TODO: how does the graph change when concatenating adding min/max

    # https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/46591.pdf

    return Vectors.dense(np.mean(neighbors, axis=0))





def create_paper_embedding(citations, embedding):

    paper_embedding = (

        citations.selectExpr("paper_id", "dst_citation_id as citation_id")

        .join(embedding, on="citation_id")

        .groupBy("paper_id")

        .agg(F.collect_list("vector").alias("vectors"))

        .withColumn("vector", udf_average_embedding("vectors"))

        .withColumn("outDegree", F.size("vectors"))

        .drop("vectors")

    )

    return paper_embedding



spark = (

    SparkSession.builder

    .config("spark.driver.memory", "12g")

    .getOrCreate()

)

spark.conf.set("spark.sql.shuffle.partitions", spark.sparkContext.defaultParallelism*2)



# create embeddings

citations = spark.read.parquet("/kaggle/input/cord-19-citation-network/citations.parquet")

%time citation_embedding = create_embedding(citations)

paper_embedding = create_paper_embedding(citations, citation_embedding)



# write out citation embedding

df = citation_embedding.toPandas()

df.vector = df.vector.apply(np.array)

df.to_parquet("citation_embedding.parquet")

print("wrote citation_embedding.parquet")



# write out paper embeddings (single iteration of aggregate neighbors)

df = paper_embedding.select("paper_id", "vector", "outDegree").toPandas()

df.vector = df.vector.apply(np.array)

df.to_parquet("paper_embedding.parquet")

print("wrote paper_embedding.parquet")
# spark.stop()

# ! killall -9 java
# metadata = pd.read_parquet("/kaggle/input/cord19-parquet/metadata.parquet")

paper_embedding = pd.read_parquet("paper_embedding.parquet")

paper_embedding
thematic_tagging = pd.read_csv("/kaggle/input/covid-19-thematic-tagging-with-regular-expressions/thematic_tagging_output_full.csv")

thematic_tagging.columns

risk = thematic_tagging[thematic_tagging.has_full_text][["sha", "tag_risk_generic"]].rename(columns={"sha": "paper_id"}).set_index("paper_id")

df = paper_embedding.set_index("paper_id").join(risk, on="paper_id")

df
import umap

import umap.plot



umap.plot.output_notebook()

%time mapper = umap.UMAP(n_neighbors=25, min_dist=0.5).fit(np.stack(df.vector))
%time mapper = umap.UMAP(n_neighbors=30, min_dist=0.4).fit(np.stack(df.vector))

p = umap.plot.diagnostic(mapper, diagnostic_type="pca")
hover_data = pd.DataFrame({

    'index': np.arange(df.shape[0]),

    'label': df.tag_risk_generic.fillna(False).values.astype(int)

})

p = umap.plot.interactive(mapper, point_size=2, labels=hover_data["label"], hover_data=hover_data)

umap.plot.show(p)
import seaborn as sns



sns.pairplot(pd.DataFrame(np.stack(df.vector)[:,:3]));