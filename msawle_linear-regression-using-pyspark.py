# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Installing PySpark package

! pip install pyspark
from pyspark.sql import SparkSession

spark = SparkSession\
        .builder\
        .appName("Python Spark regression example")\
        .config("spark.some.config.option", "some-value")\
        .getOrCreate()


df = spark.read.format("csv").\
                options(inferSchema = True, header = True).\
                load("../input/advertisingcsv/Advertising.csv", header=True)


print(df.show(5, False)) # Shows first five records and does not truncate values

print(df.printSchema())
# Describe the dataset, the result will be shown for all feature columns
df.describe().show()

# Describe the numeriacl statistics for just a signle colums

df.select("TV").describe().show()
# Printitng the column names of the dataframe
col_names = df.columns
print(col_names)
#Renaming the column "_c0" to "Index" by using withcolumnRenamed method

df = df.withColumnRenamed("_c0", "Index")

# Printing to show the change in column name
df.show(5)
from pyspark.ml.feature import RFormula

supervised = RFormula(formula = "Sales ~  TV + Radio + Newspaper")
fittedRF = supervised.fit(df)
preparedDF = fittedRF.transform(df)
preparedDF.show(5, False)
# Same above transformation can be obtained ina sig

testDf = RFormula(formula = "Sales ~  TV + Radio + Newspaper").fit(df).transform(df)

testDf.show(5, False)
from pyspark.ml.feature import VectorAssembler

va = VectorAssembler()\
    .setInputCols(["TV","Radio", "Newspaper"])\
    .setOutputCol("va_features")

va.transform(preparedDF).show(5)
# We are defining a functions that takes in a dataframe and converts it to RDD
# applies a Map function
# The function passed onto Map function returs All the columns except first & last cloumn to prepare the feature vector
# and only the last columns ("Sales") as the label vector

from pyspark.ml.linalg import Vectors

def vecTransform (data):
    return data.rdd.map(lambda r: [Vectors.dense(r[:-1]),r[-1]]).toDF(['features','label'])



vecTransform(df).show(5, False)

from pyspark.ml.feature import StandardScaler


scaler = StandardScaler().setInputCol("features").setOutputCol("scaledFeatures")

scaledDF = scaler.fit(preparedDF).transform(preparedDF)
scaledDF.show(5, False)
preparedDF.show(5)
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator

featureIndexer = VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(preparedDF)

data = featureIndexer.transform(preparedDF)

data.show(5)
# Scaling dataset

scaler = StandardScaler().setInputCol("indexedFeatures").setOutputCol("scaledFeatures")

finalScaledData = scaler.fit(data).transform(data)

finalScaledData.show(5, False)
inputData = finalScaledData.select(["scaledFeatures", "label"])

# inputData = inputData.withColumnRenamed("scaledFeatures", "features")

inputData.show(5, False)
# Spliting dataset into training and test set

training, test = inputData.randomSplit([0.8,0.2])

print("Training set has {} records".format(training.count()))
print("Training set has {} records".format(test.count()))
# Creating an instace of LinearRegression class passing on parameters

lin_reg = LinearRegression(featuresCol="scaledFeatures", labelCol = "label").setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)

print(lin_reg.explainParams()) # Explains the parameters of the model

model = lin_reg.fit(training) # Fitting the Model


print('###' * 30)
print("Coffiecients of the model are: {}".format(model.coefficients))
print('###' * 30)
print('###' * 30)

print("Intercept of linear regression equation is: {}".format(model.intercept))

print('###' * 30)
print('###' * 30)
summary = model.summary # creating an instance of model summary

print('###' * 30)
print('###' * 30)
print("The residual values after the fit are: ")
summary.residuals.show() # This returns only the first 20 values

print('###' * 30)
print('###' * 30)
print("The calculated loss for the objective function at each iteration is: ")
print(summary.objectiveHistory)

print('###' * 30)
print('###' * 30)
print("The RMSE value of the fitted model: {}".format(summary.rootMeanSquaredError))

print('###' * 30)
print('###' * 30)
print("The R2 score of the fitted model: {}".format(summary.r2))

predictions = model.transform(test)
predictions.show(30, False)
from pyspark.ml.evaluation import RegressionEvaluator

reg_eval = RegressionEvaluator(predictionCol = "prediction", labelCol = "label", metricName = "rmse")

print("The RMSE value on test data: {}".format(reg_eval.evaluate(predictions)))


from pyspark.ml import Pipeline # Importing Pipeline


train, test = df.randomSplit([0.8, 0.2]) # splitting datatset into train and test set
assembler = VectorAssembler(inputCols = ["TV", "Radio","Newspaper"], outputCol ="features")
indexer = VectorIndexer(inputCol ="features", outputCol = "indexed_features")
scaler = StandardScaler(inputCol = "indexed_features", outputCol = "scaledFeatures")
lr = LinearRegression(featuresCol = "scaledFeatures", labelCol = "Sales")

pipeline = Pipeline(stages = [assembler, indexer, scaler, lr])

model = pipeline.fit(train)
predictions = model.transform(test)
predictions.show(10, False)