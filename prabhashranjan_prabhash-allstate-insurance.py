!pip install --quiet sparkmagic
!pip install --quiet pyspark
!pyspark --version

#  Increase the width of notebook to display all columns of data

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

# Show multiple outputs of a single cell

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, SQLContext

from pyspark.sql.types import *
import pyspark.sql.functions as F
from pyspark.sql.functions import udf, col
spark = SparkSession \
            .builder.master('local[*]')\
            .appName('allstate_claims')\
            .getOrCreate()     
spark

sc = spark.sparkContext
sc
sqlContext = SQLContext(spark.sparkContext)
sqlContext
#  Read, transform and understand the data
#    pyspark creates a spark-session variable: spark

df = spark.read.csv(
                   path = "../input/allstate-claims-severity/train.csv",   
                   header = True,
                   inferSchema= True,           # Infer datatypes automatically
                   sep=","
                   )
df.take(2)
df.show(2)
df.dtypes

# Data shape
df.count()     #How many rows?      
cols = df.columns
len(cols)            
print(cols)
df.printSchema()
# What is the nature of df:
type(df)                     # pyspark.sql.dataframe.DataFrame
#  We also cache the data so that we only read it from disk once.
df.cache()
df.is_cached            # Checks if df is cached
# Show database in parts:
df.select(cols[:15]).show(3)
df.select(cols[15:25]).show(3)
df.select(cols[25:35]).show(3)
df.select(cols[35:45]).show(3)
df.select(cols[45:]).show(3)
df.tail(2)
(df.describe().select(
                    "summary",
                    F.round("cont1", 4).alias("cont1"),
                    F.round("cont2", 4).alias("cont2"),
                    F.round("cont3", 4).alias("cont3"),
                    F.round("cont4", 4).alias("cont4"),
                    F.round("cont5", 4).alias("cont5"),
                    F.round("cont6", 4).alias("cont6"),
                    F.round("cont7", 4).alias("cont7"),
                    F.round("cont13", 4).alias("cont13"),
                    F.round("cont14", 4).alias("cont14"),
                    F.round("loss", 4).alias("loss"))
                    .show())
# Adjust the values of `medianHouseValue`
df = df.withColumn("loss", col("loss")/1000)
df.show(2)
#  Which columns to drop?

columns_to_drop = ['id']
df= df.drop(*columns_to_drop)
df.dtypes
from pyspark.sql.functions import col

df.select(col("loss")).show(5)
df.select("loss").show(5)

df = df.withColumnRenamed('loss', 'label')
print(df.columns)
# setting random seed for notebook reproducability
import pandas as pd
import numpy as np

rnd_seed=23
np.random.seed=rnd_seed
np.random.set_state=rnd_seed
# Data splitting  #

# Split the dataset randomly into 70% for training and 30% for testing.
train, validation = df.randomSplit([0.7, 0.3],seed=rnd_seed)


print(train.count()/df.count())
print(validation.count()/df.count())
# Split the dataset randomly into 70% for training and 30% for testing.

#splits = df.randomSplit([0.7, 0.3])
#train = splits[0]
#test = splits[1].withColumnRenamed("loss", "Label")
#train_rows = train.count()
#test_rows = test.count()
#print("Training Rows:", train_rows, " Testing Rows:", test_rows)
train.count()
train.explain(extended=True)

#train.checkpoint()
train = spark.createDataFrame(train.rdd, schema=train.schema)

# Now, check the size of your DAG

# Displays the  length of physical plan
train.explain(extended=True)

validation.explain(extended=True)
#validation.checkpoint()
validation = spark.createDataFrame(validation.rdd, schema=validation.schema)
validation.explain(extended=True)

#  Encode 'string' column to index-column. 
#     Indexing begins from 0.
from pyspark.ml.feature import StringIndexer

# List all categorical columns and create objects to StringIndex all these categorical columns

cat_columns = [ c[0] for c in df.dtypes if c[1] == "string"]


stringindexer_stages = [ StringIndexer(inputCol=c, outputCol='stringindexed_' + c) for c in cat_columns]
stringindexer_stages
len(stringindexer_stages)
# Prepare (one) object to OneHotEncode categorical columns (received from above)
#  OHE an indexed column after StringIndexing and create one another column
from pyspark.ml.feature import OneHotEncoder

in_cols = ['stringindexed_' + c for c in cat_columns]
ohe_cols = ['onehotencoded_' + c  for c in cat_columns]
onehotencoder_stages = [OneHotEncoder(inputCols=in_cols, outputCols=ohe_cols)]
# iii)  Prepare a (one) list of all numerical and OneHotEncoded columns. Exclude 'loss' column from this list.

# Unlike in other languages, in spark
#       type-classes are to be separateky imported
#       They are not part of core classes or modules
from pyspark.sql.types import DoubleType

double_cols =   [  i[0] for i in df.dtypes if i[1] == 'double' ] 

double_cols.remove('label')  

double_cols


#  Create a combined list of double + ohe_cols

featuresCols = double_cols + ohe_cols
print(featuresCols)
len(featuresCols)

# Create a VectorAssembler object to assemble all the columns as above
from pyspark.ml.feature import VectorAssembler
#   Create an instance of VectorAssembler class.
#          This object will be used to assemble all featureCols
#          (a list of columns) into one column with name
#           'rawFeatures'

vectorassembler = VectorAssembler(
                                  inputCols=featuresCols,
                                  outputCol="rawFeatures"
                                 )
from pyspark.ml import Pipeline
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator
# Create an object to perform modeling using GBTRegressor

gbt = GBTRegressor(labelCol="label",featuresCol="rawFeatures",predictionCol='predlabel', maxIter=10)
# 9.2 Create pipeline model
pipeline = Pipeline(stages=[                        \
                             *stringindexer_stages, \
                             *onehotencoder_stages, \
                             vectorassembler,       \
                             gbt                    \
                           ]                        \
                   )
#  Run the pipeline
import os, time

start = time.time()
pipelineModel = pipeline.fit(train)
end = time.time()
(end - start)/60           

# Make predictions on validation data.
#      Note it is NOT pipelineModel.predict()

prediction = pipelineModel.transform(validation)
predicted = prediction.select("predlabel", "label")
#predicted.show(100, truncate=False)
#  Show 10 columns including predicted column
predicted.show(10, truncate=False)

predicted

# 10.3 Evaluate results
# Create evaluator object.  class is, as:

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RegressionMetrics


evaluator = RegressionEvaluator(predictionCol='predlabel', labelCol='label', metricName='rmse')

print("RMSE: {0}".format(evaluator.evaluate(predicted)))

