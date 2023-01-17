! pip install pyspark
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from pyspark.ml.linalg import Vectors

from pyspark.ml.classification import LogisticRegression

from pyspark.sql import SparkSession
help(Vectors)
spark = SparkSession.builder.master("local[*]").getOrCreate()
training = spark.createDataFrame([

0    (1.0, Vectors.dense([0.0, 1.1, 0.1])),

    (0.0, Vectors.dense([2.0, 1.0, -1.0])),

    (0.0, Vectors.dense([2.0, 1.3, 1.0])),

    (1.0, Vectors.dense([0.0, 1.2, -0.5]))], ["label", "features"])

print(training.printSchema())

training.toPandas()
lr = LogisticRegression(maxIter = 10, regParam= 0.01)
help(lr)
lr.explainParams()
model = lr.fit(training)
model.extractParamMap()
test = spark.createDataFrame([

    (1.0, Vectors.dense([-1.0, 1.5, 1.3])),

    (0.0, Vectors.dense([3.0, 2.0, -0.1])),

    (1.0, Vectors.dense([0.0, 2.2, -1.5]))], ["label", "features"])

test.toPandas()



# test_features = spark.createDataFrame([

#     (Vectors.dense([-1.0, 1.5, 1.3])),

#     (Vectors.dense([3.0, 2.0, -0.1])),

#     (Vectors.dense([0.0, 2.2, -1.5]))], ["features"])

# test_features.toPandas()
prediction = model.transform(test)
print(prediction.printSchema())

prediction.toPandas()
help(model.transform)