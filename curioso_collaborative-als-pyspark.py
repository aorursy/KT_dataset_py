# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pip install pyspark
from pyspark.context import SparkContext

from pyspark.sql.session import SparkSession

from pyspark.sql.types import *

from pyspark.ml.recommendation import ALS

from pyspark.ml.evaluation import RegressionEvaluator
sc = SparkContext('local')

spark = SparkSession(sc)
input_schema = StructType([

    StructField('userID',IntegerType(), False),

    StructField('songID',IntegerType(), False),

    StructField('rating',IntegerType(), False),

])

data = spark.read.csv(

    '../input/dataset-for-collaborative-filters/songsDataset.csv', header=True, schema=input_schema

).cache()
(training, test) = data.randomSplit([0.78, 0.22])
als = ALS(maxIter=10, regParam=0.01, userCol="userID", itemCol="songID", ratingCol="rating",

          coldStartStrategy="drop")

model = als.fit(training)
predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",

                                predictionCol="prediction")
rmse = evaluator.evaluate(predictions)

print("Root-mean-square error = " + str(rmse))