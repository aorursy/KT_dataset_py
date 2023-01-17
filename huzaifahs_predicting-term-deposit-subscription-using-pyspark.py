# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install pyspark
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("sample_app").getOrCreate()

df = spark.read.load("../input/bank-dataset/bank.csv",format="csv",inferSchema= True,header=True,sep=";")

df = df.withColumnRenamed("y","deposit")

df.printSchema()
numeric_features = [t[0] for t in df.dtypes if t[1] == 'int']

df.select(numeric_features).describe().toPandas().transpose()
numeric_data = df.select(numeric_features).toPandas()

axs = pd.plotting.scatter_matrix(numeric_data, figsize=(8, 8));

n = len(numeric_data.columns)

for i in range(n):

    v = axs[i, 0]

    v.yaxis.label.set_rotation(0)

    v.yaxis.label.set_ha('right')

    v.set_yticks(())

    h = axs[n-1, i]

    h.xaxis.label.set_rotation(90)

    h.set_xticks(())
df = df.select('age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'deposit')

cols = df.columns

df.printSchema()
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler

categoricalColumns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome']

stages = []

for categoricalCol in categoricalColumns:

    stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index')

    encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])

    stages += [stringIndexer, encoder]

label_stringIdx = StringIndexer(inputCol = 'deposit', outputCol = 'label')

stages += [label_stringIdx]

numericCols = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']

assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols

assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")

stages += [assembler]
from pyspark.ml import Pipeline

pipeline = Pipeline(stages = stages)

pipelineModel = pipeline.fit(df)

df = pipelineModel.transform(df)

selectedCols = ['label', 'features'] + cols

df = df.select(selectedCols)

df.printSchema()
train, test = df.randomSplit([0.7, 0.3], seed = 2018)

print("Training Dataset Count: " + str(train.count()))

print("Test Dataset Count: " + str(test.count()))
from pyspark.ml.classification import RandomForestClassifier



rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'label' )

rfModel = rf.fit(train)

predictions_rf = rfModel.transform(test)

predictions_rf.select('age', 'job', 'label', 'rawPrediction', 'prediction', 'probability').toPandas()

from pyspark.ml.evaluation import BinaryClassificationEvaluator

eval = BinaryClassificationEvaluator()

eval.evaluate(predictions_rf,{eval.metricName: 'areaUnderROC'})