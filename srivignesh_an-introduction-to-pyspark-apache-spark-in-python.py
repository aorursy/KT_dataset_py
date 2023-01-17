'''Install pyspark'''

!pip install pyspark
import math

import numpy as np 

import pandas as pd  

import pyspark

from pyspark.sql import SparkSession

from pyspark.sql.functions import isnan, when, count, col, isnull, asc, desc, mean



'''Create a spark session'''

spark = SparkSession.builder.master("local").appName("DataWrangling").getOrCreate()

'''Set this configuration to get output similar to pandas'''

spark.conf.set('spark.sql.repl.eagerEval.enabled', True)
df = spark.read.csv('../input/titanic/train.csv',header=True)

df.limit(5)
'''Find the count of a dataframe'''

df.count()
df.groupBy('Sex').count()
df.select('Embarked').distinct()
'''Select a single column'''

df.select('Survived').limit(2)
df.select('Survived', 'Age', 'Ticket').limit(5)
'''Find the count of missing values'''

df.select([count(when(isnull(column), column)).alias(column) for column in df.columns])
'''Find not null values of 'Age' '''

df.filter(col('Age').isNotNull()).limit(5)
'''Another way to find not null values of 'Age' '''

df.filter("Age is not NULL").limit(5)
'''Find the null values of 'Age' '''

df.filter(col('Age').isNull()).limit(5)
'''Another way to find null values of 'Age' '''

df.filter("Age is NULL").limit(5)
'''Find the mean of the column "Age" '''

mean_ = df.select(mean(col('Age'))).take(1)[0][0]

mean_ = math.ceil(mean_)
'''Find the value counts of Cabin and select the mode'''

df.groupBy(col('Cabin')).count().sort(desc("count")).limit(5)
'''Find the mode of'''

embarked_mode = df.groupBy(col('Embarked')).count().sort(desc("count")).take(1)[0][0]
'''Fill the missing values'''

df = df.fillna({'Age':mean_,'Cabin':'C23','Embarked':embarked_mode})
'''Drop a single column'''

df.drop('Age').limit(5)
'''Drop multiple columns'''

df.drop('Age', 'Parch','Ticket').limit(5)
'''Sort age in descending order'''

df.sort(desc('Age')).limit(5)
'''Sort "Parch" column in ascending order and "Age" in descending order'''

df.sort(asc('Parch'),desc('Age')).limit(5)
'''Finding the mean age of male and female'''

df.groupBy('Sex').agg(mean('Age'))
'''Finding the mean Fare of male and female'''

df.groupBy('Sex').agg(mean('Fare'))