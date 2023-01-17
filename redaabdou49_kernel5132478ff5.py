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
from pyspark.ml.evaluation import RegressionEvaluator

from pyspark.ml.recommendation import ALS

from pyspark.sql import Row

from pyspark import SparkContext

from pyspark.sql.functions import *

from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder

from pyspark.sql import SparkSession

from pyspark.mllib.regression import LabeledPoint

import numpy as np

import scipy.sparse as sps

from pyspark.mllib.linalg import Vectors

from pyspark.mllib.feature import HashingTF

from pyspark.mllib.classification import LogisticRegressionWithSGD

from pyspark.mllib.classification import SVMWithSGD, SVMModel, NaiveBayes

from pyspark.mllib.evaluation import BinaryClassificationMetrics

import pandas as pd

from scipy import stats

from pyspark.mllib.evaluation import MulticlassMetrics
spark = SparkSession.Builder().getOrCreate()
data = spark.read.csv("/kaggle/input/predicting-a-pulsar-star/pulsar_stars.csv",header=True,inferSchema=True)

data.limit(5).toPandas()
data = data.withColumnRenamed(" Excess kurtosis of the integrated profile","target_reg")
data.limit(5).toPandas()
from pyspark.ml.feature import VectorAssembler



columns = [col for col in data.columns if col not in ['target_reg']]



assembler = VectorAssembler().setInputCols(columns).setOutputCol("features")

train = assembler.transform(data)
train.limit(5).toPandas()
from pyspark.ml.regression import RandomForestRegressor

from pyspark.ml.evaluation import RegressionEvaluator





# Split the data into training and test sets (30% held out for testing)

(trainingData, testData) = train.randomSplit([0.8, 0.2])



# Train a RandomForest model.

rf = RandomForestRegressor(featuresCol="features",labelCol = "target_reg")





# Train model.  This also runs the indexer.

model = rf.fit(trainingData)

# Make predictions.

predictions = model.transform(testData)



# Select example rows to display.

predictions.select("prediction", "target_reg", "features").show(5)



# Select (prediction, true label) and compute test error

evaluator = RegressionEvaluator(

    labelCol="target_reg", predictionCol="prediction", metricName="rmse")

rmse = evaluator.evaluate(predictions)

print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)