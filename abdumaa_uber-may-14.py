from pyspark import SparkContext
from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local[*]").getOrCreate()
!wget -q https://raw.githubusercontent.com/fivethirtyeight/uber-tlc-foil-response/master/uber-trip-data/uber-raw-data-may14.csv
from pyspark.sql.types import *
schema = StructType([
                     StructField("time", TimestampType(), True),
                     StructField("lat", DoubleType(), True),
                     StructField("lon", DoubleType(), True),
                     StructField("base", StringType(), True)
                     ])
df = spark.read.csv("uber-raw-data-may14.csv", header = "True" ,schema= schema, mode="DROPMALFORMATED", sep=",", timestampFormat = "MM/dd/yyyy HH:mm:ss" )
from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols=['lat', 'lon'], outputCol = 'features')
featureDf = assembler.transform(df)
featureDf.printSchema()
featureDf.show(10)
from pyspark.ml.clustering import GaussianMixture

gmm = GaussianMixture().setK(8)
gmmmodel = gmm.fit(featureDf)
gmmpredictDf = gmmmodel.transform(featureDf)
from pyspark.ml.evaluation import ClusteringEvaluator
evaluator = ClusteringEvaluator()
silhouette = evaluator.evaluate(gmmpredictDf)
print("GMM Silhouette with squared euclidean distance = " + str(silhouette))
import matplotlib.pyplot as plt
plt.title("GMM")
plt.scatter(gmmpredictDf.select('lat').collect(),gmmpredictDf.select('lon').collect(),c=gmmpredictDf.select('prediction').collect(), marker='.')
from pyspark.ml.clustering import KMeans

kmeans = KMeans(maxIter=8000).setK(8).setFeaturesCol("features").setPredictionCol("prediction")

kmeansModel = kmeans.fit(featureDf)
predictDf = kmeansModel.transform(featureDf)
predictDf.show(10)
# Evaluate clustering by computing Silhouette score
from pyspark.ml.evaluation import ClusteringEvaluator
evaluator = ClusteringEvaluator()

silhouette = evaluator.evaluate(predictDf)
print("Silhouette with squared euclidean distance = " + str(silhouette))
import matplotlib.pyplot as plt
plt.title("K-Means")
plt.scatter(predictDf.select('lat').collect(),predictDf.select('lon').collect(),c=predictDf.select('prediction').collect(), marker='.')