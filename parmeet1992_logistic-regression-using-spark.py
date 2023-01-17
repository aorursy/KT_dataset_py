# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/test"))

#!unzip ../input/test/

# Any results you write to the current directory are saved as output.

!pip install pyspark
import pandas as pd

from pyspark.sql import SparkSession,SQLContext

from pyspark import SparkFiles

from pyspark.ml.feature import VectorAssembler

import pyspark

from pyspark.ml.feature import StringIndexer

from pyspark.ml import Pipeline

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

sc = pyspark.SparkContext('local[*]')
sqlCtx = SQLContext(sc)

df = sqlCtx.read.csv('../input/train/train.csv',header=True,inferSchema='True')

df_test = sqlCtx.read.csv('../input/test/test.csv',header=True,inferSchema='True')
spark = SparkSession.builder.appName("pet_adoption").getOrCreate()

##pandas frame is easier to read

df_pd = pd.read_csv('../input/train/train.csv')

input_cols = [a for a,b in df.dtypes if b=='int']
indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(df) for column in ["AdoptionSpeed"]]

pipeline = Pipeline(stages=indexers)

df = pipeline.fit(df).transform(df)

df_test = pipeline.fit(df_test).transform(df_test)



feature = VectorAssembler(inputCols=input_cols,outputCol="features")

feature_vector= feature.transform(df)



feature_vector_test= feature.transform(df_test)

(trainingData, testData) = feature_vector.randomSplit([0.8, 0.2],seed = 11)

from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(labelCol="AdoptionSpeed_index", featuresCol="features")

lrModel = lr.fit(trainingData)

lr_prediction = lrModel.transform(testData)

#lr_prediction.select("prediction", "Survived", "features").show()

#evaluator = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="accuracy")

evaluator = MulticlassClassificationEvaluator(labelCol="AdoptionSpeed_index", predictionCol="prediction", metricName="accuracy")

lr_accuracy = evaluator.evaluate(lr_prediction)

print("Accuracy of LogisticRegression is = %g"% (lr_accuracy))

print("Test Error of LogisticRegression = %g " % (1.0 - lr_accuracy))

#lr_prediction.show()

lr_prediction = lrModel.transform(feature_vector_test)

predictions = [int(elem['prediction']) for elem in lr_prediction.select('prediction').collect()]

predictions_ids = [elem['PetID'] for elem in lr_prediction.select('PetID').collect()]

df_new = pd.DataFrame()

df_new['PetID'] = predictions_ids

df_new['AdoptionSpeed'] = predictions

df_new.to_csv('submission.csv',index=False)