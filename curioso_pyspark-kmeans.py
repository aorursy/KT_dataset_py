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

from pyspark.ml.feature import VectorAssembler

from pyspark.ml.clustering import KMeans

import numpy as np

import matplotlib.pyplot as plt

from pyspark.ml.evaluation import ClusteringEvaluator
sc = SparkContext('local')

spark = SparkSession(sc)
input_schema = StructType([

    StructField('userID',IntegerType(), False),

    StructField('songID',IntegerType(), False),

])

data = spark.read.csv(

    '../input/dataset-for-collaborative-filters/songsDataset.csv', header=True, schema=input_schema

).cache()
data.printSchema()
data.show(10)
vecAssembler = VectorAssembler(inputCols=['userID', 'songID'], outputCol="features")

data = vecAssembler.transform(data).select('features')

data.show()
cost = np.zeros(22)

for k in range(2,22):

    kmeans = KMeans().setK(k).setSeed(1).setFeaturesCol("features")

    model = kmeans.fit(data.sample(False,0.1))

    cost[k] = model.computeCost(data)
fig, ax = plt.subplots(1,1, figsize =(10,7))

ax.plot(range(2,22),cost[2:22])

ax.set_xlabel('k')

ax.set_ylabel('cost')