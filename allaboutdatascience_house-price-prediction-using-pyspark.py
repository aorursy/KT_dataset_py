# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load



# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#installing pyspark
!pip install pyspark
#Importing required libraries 
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql import SparkSession

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.regression import LinearRegression
#creating a spark session
spark = SparkSession.builder.master("house_price").getOrCreate()
#reading csv files
train=spark.read.csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv", inferSchema=True, header=True)
test=spark.read.csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv", inferSchema=True, header=True)
train.printSchema()
#viewing first 3 lines using head() and looping to make it readable
for house in train.head(3):
    print(house)
    print("\n")
#String Indexer are used tranform strings into categorical data. We are doing it for only one column here but we can doit for all string data
indexer= StringIndexer(inputCol="LotShape", outputCol="LotShape2")
indexed= indexer.fit(train).transform(train)
indexed.head(1)
#Assembler combines all integer and create a vector which is used as input to predict. Here we have only selected columns with data type as integer
assembler= VectorAssembler(inputCols=["MSSubClass","LotArea","OverallQual","OverallCond","BsmtFinSF1",
                                      "BsmtFinSF2","BsmtUnfSF","TotalBsmtSF","1stFlrSF","2ndFlrSF","LowQualFinSF","GrLivArea","BsmtFullBath","BsmtHalfBath",
                                     "FullBath","HalfBath","BedroomAbvGr","KitchenAbvGr","TotRmsAbvGrd","Fireplaces","YearBuilt",
                                     "YearRemodAdd","GarageCars","GarageArea","WoodDeckSF","OpenPorchSF","EnclosedPorch","3SsnPorch","ScreenPorch","PoolArea",
                                     "MiscVal","MoSold","YrSold","LotShape2"],outputCol="features")
#transforming assembler
output= assembler.transform(indexed)
output.select("features","SalePrice")
#We can see column features is dense vector
final=output.select("features","SalePrice")
final.head(3)
#We will split data into train and validate
train_df,valid_df= final.randomSplit([0.7,0.3])
train_df.describe().show()
#initializing and fitting model
lr= LinearRegression(labelCol="SalePrice")
model= lr.fit(train_df)
#fitting model of validation set
validate=model.evaluate(valid_df)
#let's check how model performed
print(validate.rootMeanSquaredError)
print(validate.r2)