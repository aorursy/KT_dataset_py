!pip install pyspark
from pyspark.sql import SparkSession

spark_ex = SparkSession.builder.getOrCreate()

print(spark_ex)
# Don't change this file path

file_path = "../input/titanic/train.csv"



# Read in the titanic data

titanic = spark_ex.read.csv(file_path,header=True)



# Show the data

titanic.show()
titanic=titanic.drop("Name", "Ticket","Cabin")

titanic.show()
titanic=titanic.filter(titanic.Pclass.isNotNull())

titanic=titanic.filter(titanic.Sex.isNotNull())

titanic=titanic.filter(titanic.Age.isNotNull())

titanic=titanic.filter(titanic.SibSp.isNotNull())

titanic=titanic.filter(titanic.Fare.isNotNull())

titanic=titanic.filter(titanic.Parch.isNotNull())

titanic=titanic.filter(titanic.Embarked.isNotNull())

titanic.show()
#total number of final records with us

titanic.count()
titanic.printSchema()
#casting columns to numeric

titanic = titanic.withColumn("PassengerId", titanic.PassengerId.cast("integer"))

titanic = titanic.withColumn("label", titanic.Survived.cast("integer"))

titanic = titanic.withColumn("Age", titanic.Age.cast("integer"))

titanic = titanic.withColumn("SibSp", titanic.SibSp.cast("integer"))

titanic = titanic.withColumn("Parch", titanic.Parch.cast("integer"))

titanic = titanic.withColumn("Fare", titanic.Fare.cast("integer"))
titanic.printSchema()
from pyspark.ml.feature import StringIndexer,OneHotEncoder



# Encoding Categorical Features

Pclass_indexer = StringIndexer(inputCol="Pclass",outputCol="Pclass_index")



# Create a OneHotEncoder

Pclass_encoder = OneHotEncoder(inputCol="Pclass_index",outputCol="Pclass_fact")
# Encoding Categorical Features

Sex_indexer = StringIndexer(inputCol="Sex",outputCol="Sex_index")



# Create a OneHotEncoder

Sex_encoder = OneHotEncoder(inputCol="Sex_index",outputCol="Sex_fact")
# Encoding Categorical Features

Embarked_indexer = StringIndexer(inputCol="Embarked",outputCol="Embarked_index")



# Create a OneHotEncoder

Embarked_encoder = OneHotEncoder(inputCol="Embarked_index",outputCol="Embarked_fact")
titanic.printSchema()
from pyspark.ml.feature import VectorAssembler

vec_assembler = VectorAssembler(inputCols=["Pclass_fact", "Sex_fact", "Age", "SibSp", "Parch","Fare","Embarked_fact"], outputCol="features")
# Import Pipeline

from pyspark.ml import Pipeline



# Make the pipeline

titanic_pipe = Pipeline(stages=[Pclass_indexer, Pclass_encoder, Sex_indexer, Sex_encoder,Embarked_indexer, Embarked_encoder, vec_assembler])
# Fit and transform the data

piped_data = titanic_pipe.fit(titanic).transform(titanic)
from pyspark.sql.functions import *

# Split the data into training and test sets

training, test = piped_data.randomSplit([.7,.3])
# Import LogisticRegression

from pyspark.ml.classification import LogisticRegression



# Create a LogisticRegression Estimator

lr = LogisticRegression()
# Import the evaluation submodule

import pyspark.ml.evaluation as evals



# Create a BinaryClassificationEvaluator

evaluator = evals.BinaryClassificationEvaluator(metricName="areaUnderROC")
# Import the tuning submodule

import numpy as np

import pyspark.ml.tuning as tune



# Create the parameter grid

grid = tune.ParamGridBuilder()



# Add the hyperparameter

grid = grid.addGrid(lr.regParam, np.arange(0, .1, .01))

grid = grid.addGrid(lr.elasticNetParam, [0,1])



# Build the grid

grid = grid.build()
# Create the CrossValidator

cv = tune.CrossValidator(estimator=lr,

               estimatorParamMaps=grid,

               evaluator=evaluator

               )
# Fit cross validation models

models = cv.fit(training)



# Extract the best model

best_lr = models.bestModel

print(best_lr)
# Use the model to predict the test set

test_results = best_lr.transform(test)



# Evaluate the predictions

print(evaluator.evaluate(test_results))