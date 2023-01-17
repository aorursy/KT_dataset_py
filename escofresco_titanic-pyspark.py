# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Pyspark setup
!pip install pyspark
import pyspark.sql.functions as sql_func
from pyspark.sql.types import *
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics
from pyspark.ml.evaluation import RegressionEvaluator

sc = SparkContext('local') #https://stackoverflow.com/questions/30763951/spark-context-sc-not-defined
spark = SparkSession(sc)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_train = pd.read_csv('../input/titanic/train.csv')
df_gender = pd.read_csv('../input/titanic/gender_submission.csv')
df_test = pd.read_csv('../input/titanic/test.csv')
print(f'train: {len(df_train)} test: {len(df_test)} df_gender: {len(df_gender)}')
df_train.dtypes
data_schema = StructType([
    StructField(column, 
                StringType() if df_train[column].dtype == object else (IntegerType() 
                                                                      if df_train[column].dtype == int else FloatType()), False)
    for column in df_train.columns])
spark_train = spark.read.csv(
    '../input/titanic/train.csv', header=True, schema=data_schema
).cache()
spark_train[list(df_train.columns)[:len(df_train.columns)//2]].show()
spark_train[list(df_train.columns)[len(df_train.columns)//2:]].show()
import matplotlib
matplotlib.pyplot.hist(list(
    filter(lambda x: x is not None, spark_train.select("age").rdd.flatMap(lambda x: x).collect()) # https://stackoverflow.com/a/40270045
))
# https://www.datasciencemadesimple.com/count-of-missing-nanna-and-null-values-in-pyspark/
from pyspark.sql.functions import isnan, when, count, col
spark_train.select([count(when(col('Age').isNull(),True))]).show() 
import pyspark.sql.functions as F

spark_train = spark_train.withColumn("Gender", F.udf(lambda s: 1 if s == 'male' else 0, IntegerType())('Sex'))
spark_train.show()
len(spark_train.columns)
list(filter(lambda x: x is not None, spark_train.select("age").rdd.flatMap(lambda x: x).collect())) # https://stackoverflow.com/a/40270045
spark_train.filter(spark_train['Embarked'] == 'C').show()
spark_train.select('Age').describe().show()
spark_train.filter(spark_train['Survived'] == 1).select('Age').describe().show()
spark_train.filter(spark_train['Survived'] == 0).select('Age').describe().show()
