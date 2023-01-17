! pip install pyspark




import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from pyspark.sql import SparkSession



spark = SparkSession.builder.getOrCreate()

spark
sdf_train = spark.read.csv("../input/train.csv",inferSchema=True,header=True)

print(sdf_train.printSchema())

pdf = sdf_train.limit(10).toPandas()

pdf.T
sdf_test = spark.read.csv("../input/test.csv",inferSchema=True,header=True)

# sdf_train.printSchema()

pdf = sdf_test.limit(10).toPandas()



pdf.T
numeric_cols = ['PassengerId','Survived', 'Pclass','Age', 'SibSp','Parch','Ticket','Fare'] 

numeric_features = ['Pclass','Age', 'SibSp','Parch','Fare'] 

from pyspark.sql import DataFrame 

from pyspark.sql import functions as F

from pyspark.ml.feature import Imputer



# default is mean

def _claenup(sdf: DataFrame,colList: list):

    imputer = Imputer(inputCols = colList,

                     outputCols = colList)

    sdf = imputer.fit(sdf).transform(sdf)

    return sdf



sdf_train_cleaned = _claenup(sdf_train,['Fare','Age'])

sdf_train_cleaned.limit(5).toPandas().T

# sdf_train_subset.select(numeric_features).printSchema() #.toPandas().T
_stages = []
# from pyspark.sql import functions as F

# sdf_train_cleaned.groupBy("Sex").agg(F.count(sdf_train_cleaned["Sex"])).show()

# sdf_train_cleaned.groupBy("Embarked").agg(F.count(sdf_train_cleaned["Embarked"])).show()
from pyspark.ml import Pipeline

from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator



string_features = [ 'Embarked', 'Sex'] # 'Cabin', 



def _strIndexer(_stages):

    strIndexer = [StringIndexer(inputCol=column, outputCol=column+"_Index", handleInvalid = "skip") for column in string_features ] #.fit(sdf_train_subset)

    _stages += strIndexer
def _oneHotEnc(_stages):

    oneHotEnc = [OneHotEncoderEstimator(inputCols= [column +"_Index"  for column in string_features]  , outputCols= [column+"_Enc" for column in string_features] )]

    _stages += oneHotEnc

from pyspark.ml.feature import VectorAssembler

# assemblerInput.clear()

def _vectAssembler(_stages):

    assemblerInput =  [f  for f in numeric_features]  

    assemblerInput += [f + "_Enc" for f in string_features]

    print(assemblerInput) 

    # assemblerInput.append("sexIndex")

    vectAssembler = VectorAssembler(inputCols  = assemblerInput, outputCol = "vect_features") #.fit(sdf_train_subset)  

    _stages += [vectAssembler]

from pyspark.ml.feature import PCA

# help(PCA)

def _pac(_stages):

    pca = PCA(inputCol = "vect_features", outputCol = "pca_features",k = 5)

    _stages += [pca]
from pyspark.ml.classification import RandomForestClassifier

# help(RandomForestClassifier)

def _rf(_stages):

    rf = RandomForestClassifier(labelCol = 'Survived', featuresCol = 'pca_features', numTrees = 100, maxDepth = 4, maxBins = 1000)

    _stages += [rf]
# _stages += [oneHotEnc, vectAssembler,rf]

_stages = []

_strIndexer(_stages)

_oneHotEnc(_stages)

_vectAssembler(_stages)

_pac(_stages)

_rf(_stages)

_stages
from pyspark.ml import Pipeline



pipeline = Pipeline(stages = _stages)
model = pipeline.fit(sdf_train_cleaned)
numeric_cols_test = ['PassengerId', 'Pclass','Age', 'SibSp','Parch','Ticket','Fare', 'Sex'] 



sdf_test_subset = sdf_test.withColumn('Ticket', sdf_test['Ticket'].cast("double"))



sdf_test_subset = _claenup(sdf_test_subset,['Fare','Age'])
sdf_predict = model.transform(sdf_test_subset)
sdf_predict.toPandas().T
# from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

from pyspark.ml.evaluation import BinaryClassificationEvaluator 



# sdf_train_sample = sdf_train_cleaned.sample(False,0.1).cache()

# # sdf_test_sample.count()

# pca_model = PCA(inputCol = "vect_features", outputCol = "pca_features",k = 5)

# cv_rf = RandomForestClassifier(labelCol = "Survived", featuresCol = "pca_features")

# cv_pipeline = Pipeline(stages = [pca_model,rf])



# param_grid = ParamGridBuilder().addGrid(pca_model.k, [10, 20, 30, 40, 50])\

#                                 .addGrid(cv_rf.numTrees, [20, 30, 50])\

#                                 .build()

# cross_val = CrossValidator(estimator = cv_pipeline,

#                           estimatorParamMaps = param_grid,

#                           evaluator = BinaryClassificationEvaluator(labelCol= 'STP_UP_IND',

#                                                                    rawPredictionCol = 'probability',

#                                                                    metricName = 'areaUnderROC'),

#                           numFolds = 3)

# cv_model = cross_val.fit(sdf_test_sample)



# cv_prediction = cv_model.transform(sdf_test_subset)



# evaluator= BinaryClassificationEvaluator(labelCol = "STP_UP_IND", rawPredictionCol="probability", metricName= "areaUnderROC")

# accuracy = evaluator.evaluate(predictions)

# accuracy

# # eval = BinaryClassificationEvaluator()

evaluator = BinaryClassificationEvaluator(labelCol = 'prediction')

evaluator.evaluate(sdf_predict)
sdf_submission = sdf_predict.select('PassengerId','prediction').withColumn('Survived',sdf_predict['prediction'].cast('integer')).select('PassengerId','Survived')

sdf_submission.toPandas().T
sdf_submission.coalesce(1).write.csv("submission",mode="overwrite",header=True)

print(os.listdir('submission'))
