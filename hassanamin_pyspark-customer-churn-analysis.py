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
#!pip install pyspark
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('churnanalysis').getOrCreate()
from pyspark.ml.classification import LogisticRegression

input_data=spark.read.csv('../input/Churn_Modelling.csv',header=True,inferSchema=True)
input_data.printSchema() #training data
from pyspark.ml.linalg import Vectors

from pyspark.ml.feature import VectorAssembler

 

assembler=VectorAssembler(inputCols=['Age','NumOfProducts','IsActiveMember','Tenure','CreditScore'],outputCol='features')

 

output_data=assembler.transform(train)
final_data=output_data.select('features','Exited')         #creating final data with only 2 columns

 

train,test=final_data.randomSplit([0.7,0.3])          #splitting data



print("Training Dataset Count: " + str(train.count()))

print("Test Dataset Count: " + str(test.count()))

 
model=LogisticRegression(labelCol='Exited')           #creating model

 

model=model.fit(train)        #fitting model on training dataset

 

summary=model.summary

 

summary.predictions.describe().show()         #summary of the predictions on training data
import matplotlib.pyplot as plt

import numpy as np

beta = np.sort(model.coefficients)

plt.plot(beta)

plt.ylabel('Beta Coefficients')

plt.show()
trainingSummary = model.summary

roc = trainingSummary.roc.toPandas()

plt.plot(roc['FPR'],roc['TPR'])

plt.ylabel('False Positive Rate')

plt.xlabel('True Positive Rate')

plt.title('ROC Curve')

plt.show()

print('Training set areaUnderROC: ' + str(trainingSummary.areaUnderROC))
pr = trainingSummary.pr.toPandas()

plt.plot(pr['recall'],pr['precision'])

plt.ylabel('Precision')

plt.xlabel('Recall')

plt.show()
predictions = model.transform(test)



predictions.select('Exited', 'rawPrediction', 'prediction', 'probability').show(10)

#summary=model.summary

 

predictions.describe().show()         #summary of the predictions on training data