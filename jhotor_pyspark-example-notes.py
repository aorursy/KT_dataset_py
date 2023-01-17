from pyspark.sql import SparkSession

import pandas as pd

spark = SparkSession.builder.appName('spark_test_notes').getOrCreate()

import os

print(os.listdir("../input"))
df = spark.read.csv("../input/metro-bike-share-trip-data.csv",inferSchema=True,header=True)
summary_df=df.describe()
summary_df.toPandas()
df.printSchema()
from pyspark.sql.functions import desc

df.groupBy("Trip ID").count().sort(desc("count")).show(5)
df.select("Duration").describe().show()
from pyspark.sql.functions import desc

df.groupBy("Starting Station ID").count().sort(desc("count")).show(5)
from pyspark.sql.functions import desc

df.groupBy("Ending Station ID").count().sort(desc("count")).show(5)
from pyspark.sql.functions import desc

df.groupBy("Bike ID").count().sort(desc("count")).show(5)

df.groupBy("Plan Duration").count().sort(desc("count")).show()
from pyspark.sql.functions import desc

df.groupBy("Trip Route Category").count().sort(desc("count")).show()

df.groupBy("Passholder Type").count().sort(desc("count")).show()
import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("darkgrid")

PT = df.groupBy("Passholder Type").count().sort(desc("count"))

PT = PT.toPandas()

sns.barplot(x="Passholder Type",y="count",data=PT)

plt.title("Distribution by Passholder Type")
sub_df = df.select("Trip ID","Starting Station ID","Ending Station ID")
sub_df.head(5)
sub_df.select("Trip ID").count()
sub_df = sub_df.dropna()

sub_df.count()
sub_df.groupBy(["Starting Station ID","Ending Station ID"]).count().sort(desc("count")).show()
sub_df =sub_df.filter(sub_df["Starting Station ID"]!=sub_df["Ending Station ID"])
sub_df = sub_df.dropDuplicates(["Starting Station ID","Ending Station ID"])

sub_df.count()
sub_df.filter(sub_df["Starting Station ID"]==sub_df["Ending Station ID"]).count()
sub_df.filter(sub_df["Starting Station ID"].between(3000,3005)).show()
sub_df = sub_df.withColumnRenamed("Trip ID","id")

sub_df.head(5)
from pyspark.sql.functions import *

sub_df = sub_df.withColumn("items",array("Starting Station ID","Ending Station ID"))
sub_df.show(5)

sub_df.printSchema()
sub_df = sub_df.select("id","items")

sub_df.show(5)

sub_df.printSchema()

sub_df.count()
sub_df = sub_df.dropDuplicates(["items"])

sub_df.count()
from pyspark.ml.fpm import FPGrowth
fpGrowth = FPGrowth(itemsCol="items", minSupport=0.0001, minConfidence=0.0001)

model = fpGrowth.fit(sub_df)
model.freqItemsets.sort(desc("freq")).show()
model.associationRules.sort(desc("confidence")).show(25)
from sklearn.datasets import load_iris
idf = load_iris()
features = pd.DataFrame(idf["data"])

labels = pd.DataFrame(idf["target"])
features.columns =(["Sepal_Lenght","Sepal_Width","Petal_Length","Petal_Width"])

labels.columns = (["label"])
iris_df = pd.concat([features,labels],axis=1)

iris_df.tail()
iris = spark.createDataFrame(iris_df)
names = iris.drop('label').columns
iris.columns
iris.show(10)
from pyspark.ml.feature import VectorAssembler

from pyspark.ml.linalg import Vectors
assembler = VectorAssembler(inputCols=["Sepal_Lenght","Sepal_Width","Petal_Length","Petal_Width"],outputCol="features")
iris = assembler.transform(iris)
iris = iris.select("label","features")

iris.show(5)
(training,testing) = iris.randomSplit([0.7, 0.3],seed=1234)

training

testing
from pyspark.ml import Pipeline

from pyspark.ml.classification import RandomForestClassifier

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
dtc = RandomForestClassifier(featuresCol="features",labelCol="label",seed=1234,numTrees=25)
pipeline = Pipeline(stages=[dtc])
dtc_model = pipeline.fit(training)
dtc_predictions = dtc_model.transform(testing)
dtc_predictions.show(5)
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(dtc_predictions)

print("Accuracy: ",accuracy)

print("Test Error = %g" % (1.0 - accuracy))
dtc_predictions.groupBy("label").pivot("prediction").count().show()