

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))

!pip install pyspark
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("vivek_spark_app").getOrCreate()

sdf = spark.read.load("../input/bank.csv",format="csv",inferSchema= True,header=True,sep=";")

# sdf.take(2)

sdf = sdf.withColumnRenamed("y","deposit")

sdf.printSchema()
pdf = sdf.toPandas()

pdf
# pd.DataFrame(sdf.take(5), columns=sdf.columns).traspose()
# Numeric features

num_features = [t[0] for t in sdf.dtypes if t[1] == 'int']

num_features
# Feature engineering



from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler

categoricalColumns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome']

stages = []



for categoricalCol in categoricalColumns:

    stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index')

    encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])

    stages += [stringIndexer, encoder]

    

label_stringIdx = StringIndexer(inputCol = 'deposit', outputCol = 'label')

stages += [label_stringIdx]

numericCols = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']

assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols

print(f"assemblerInputs : {assemblerInputs}")

assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")

print(f"assembler : {assembler}")

stages += [assembler]

print(f"stages : {stages}")
# from pyspark.ml import Pipeline

# cols = sdf.columns

# pipeline = Pipeline(stages = stages)

# pipelineModel = pipeline.fit(sdf)

# sdf = pipelineModel.transform(sdf)

# selectedCols = ['label', 'features'] + cols

# sdf = sdf.select(selectedCols)

sdf.printSchema()
print(sdf.select('features').take(2))

sdf.select('label').take(2)
train, test = sdf.randomSplit([0.7, 0.3], seed = 2018)

print("Training Dataset Count: " + str(train.count()))

print("Test Dataset Count: " + str(test.count()))

print(type(train))


from pyspark.ml.classification import DecisionTreeClassifier



dt = DecisionTreeClassifier(featuresCol = 'features', labelCol = 'label', maxDepth=3) #maxDepth to avoide overfitting



dtModel = dt.fit(train) # train the model

predictions_dt = dtModel.transform(test) # test the model / make prediction 



# pd.DataFrame( predictions_dt.select('age', 'job', 'label', 'rawPrediction', 'prediction', 'probability').take(5)
pd.DataFrame( predictions_dt.select('age', 'job', 'label', 'rawPrediction', 'prediction', 'probability').take(5),

             columns=['age', 'job', 'label', 'rawPrediction', 'prediction', 'probability']).transpose()

# predictions_dt.columns
from pyspark.ml.evaluation import BinaryClassificationEvaluator

eval = BinaryClassificationEvaluator()

eval.evaluate(predictions_dt,{eval.metricName: "areaUnderROC"})

#ROC ?
from pyspark.ml.classification import RandomForestClassifier



rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'label' )

rfModel = rf.fit(train)

predictions_rf = rfModel.transform(test)

predictions_rf.select('age', 'job', 'label', 'rawPrediction', 'prediction', 'probability').toPandas()

eval = BinaryClassificationEvaluator()

eval.evaluate(predictions_rf,{eval.metricName: 'areaUnderROC'})
from pyspark.ml.classification import GBTClassifier



gbt = GBTClassifier()

gbtModel = gbt.fit(train)

gbtPrediction =  gbtModel.transform(test)

gbtPrediction.select('age', 'job', 'label', 'rawPrediction', 'prediction', 'probability').toPandas()

eval.evaluate(gbtPrediction,{eval.metricName: "areaUnderROC"})