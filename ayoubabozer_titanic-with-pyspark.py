# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../inpreprocessing ```Age```put/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# make sure to turn on the Interenet switch in the right side menu

! pip install pyspark
from pyspark.sql import SparkSession

from pyspark.ml import Pipeline

import pyspark.sql.functions as F

from pyspark.sql.types import DoubleType, StringType, StructType, StructField

from pyspark.ml.feature import StringIndexer, VectorAssembler, QuantileDiscretizer

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from pyspark import SparkContext

from pyspark.ml.classification import RandomForestClassifier

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# build spark session

spark = SparkSession.builder.appName("Spark on titanic data ").getOrCreate()
train = spark.read.csv("/kaggle/input/titanic/train.csv", header="true", inferSchema="true")

test = spark.read.csv("/kaggle/input/titanic/train.csv", header="true", inferSchema="true")
type(train)
train.show(5)
train.printSchema()
# statistical summary

train.describe().show()
train_count = train.count()
print(train_count)
survived_groupped_df = train.groupBy("Survived").count()
survived_groupped_df.show()
getRatio = F.udf(lambda x: round(x/train_count,2), DoubleType())

survived_groupped_df = survived_groupped_df.withColumn("Ratio", getRatio('count'))
survived_groupped_df.show()
train.groupBy("Sex").count().show()
train.groupBy("Sex").agg(F.mean('Survived'), F.sum('Survived')).show()
train.createOrReplaceTempView("train")
spark.sql("SELECT Sex, round(SUM(Survived)/count(1),2) as ratio  FROM train GROUP BY Sex").show()
spark.sql("SELECT Pclass, round(SUM(Survived)/count(1),2) as ratio  FROM train GROUP BY Pclass").show()
combined = train.union(test)
combined.count()
combined.createOrReplaceTempView("combined")
null_columns = []

for col_name in combined.columns:

    null_values = combined.where(F.col(col_name).isNull()).count()

    if(null_values > 0):

        null_columns.append((col_name, null_values))

print(null_columns)
spark.createDataFrame(null_columns, ['column', 'missing_value']).show()
spark.sql("SELECT Name  FROM combined").show()
combined = combined.withColumn('Title',F.regexp_extract(F.col("Name"),"([A-Za-z]+)\.",1))
combined.createOrReplaceTempView('combined')

spark.sql("SELECT Title,count(1)  FROM combined GROUP BY Title").show()
titles_map = {

 'Capt' : 'Rare',

 'Col' : 'Rare',

 'Don': 'Rare',

 'Dona': 'Rare',

 'Dr' : 'Rare',

 'Jonkheer' :'Rare' ,

 'Lady': 'Rare',

 'Major': 'Rare',

 'Master': 'Master',

 'Miss' : 'Miss',

 'Mlle' : 'Rare',

 'Mme': 'Rare',

 'Mr': 'Mr',

 'Mrs': 'Mrs',

 'Ms': 'Rare',

 'Rev': 'Rare',

 'Sir': 'Rare',

 'Countess': 'Rare'

}

def impute_title(title):

    return titles_map[title]



title_map_func = F.udf(lambda x: impute_title(x), StringType())



combined = combined.withColumn('Title', title_map_func('Title'))
combined.createOrReplaceTempView('combined')

spark.sql("SELECT Title  FROM combined GROUP BY Title").show()
round(spark.sql("SELECT AVG(Age) FROM combined").collect()[0][0])
combined = combined.fillna(30, subset=['Age'])
groupped_Embarked = spark.sql("SELECT Embarked,count(1) as count_it FROM combined GROUP BY Embarked ORDER BY count_it DESC")
groupped_Embarked.show()
embarked_mode = groupped_Embarked.collect()[0][0]
print(embarked_mode)
combined = combined.fillna(embarked_mode, subset=['Embarked'])
combined = combined.withColumn("Cabin", combined.Cabin.substr(0, 1))
combined.createOrReplaceTempView('combined')

groupped_cabin = spark.sql("SELECT Cabin,count(1) as count_it FROM combined GROUP BY Cabin ORDER BY count_it DESC")

groupped_cabin.show()
combined = combined.fillna('U', subset=['Cabin'])
combined = combined.withColumn('Family_size', F.col('Parch')+ F.col('SibSp'))
indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(combined) for column in ["Sex","Embarked","Title", "Cabin"]]





pipeline = Pipeline(stages=indexers)

combined = pipeline.fit(combined).transform(combined)



combined.show()
combined = combined.drop('Sex','PassengerId','Name','Title', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked')
combined_pandas = combined.toPandas()
combined_pandas
train_pandas = combined_pandas[:train_count]

test_pandas = combined_pandas[train_count:]
train = spark.createDataFrame(train_pandas)

test = spark.createDataFrame(test_pandas)

test = test.drop('Survived')
assembler = VectorAssembler(inputCols=train.columns[1:],outputCol="features")

train_assembler_vector = assembler.transform(train)

train_assembler_vector.show()

test_assembler = VectorAssembler(inputCols=test.columns,outputCol="features")

test_assembler_vector = test_assembler.transform(test)

test_assembler_vector.show()

X_train, X_test = train_assembler_vector.randomSplit([0.8, 0.2],seed = 11)
rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'Survived')

rfModel = rf.fit(X_train)

predictions = rfModel.transform(X_test)

predictions.select("prediction", "Survived", "features").show()
evaluator = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="accuracy")

print("Accuracy : " + str(evaluator.evaluate(predictions)))
submission = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
test = test.drop('Survived')


final_predictions = rfModel.transform(test_assembler_vector)

final_predictions = final_predictions.toPandas()
final_predictions
submission['Survived'] = final_predictions['prediction']
submission['Survived'] = submission['Survived'].astype(int)
submission.to_csv("RF.csv",index=False)