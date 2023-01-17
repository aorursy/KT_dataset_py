# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from pyspark import SparkContext,SQLContext
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
sc=SparkContext(appName='pimaDiabetesApp')
sql=SQLContext(sc)
rawDF=sql.read.format('csv').options(header=True,inferSchema=True).load('../input/pima-indians-diabetes-deepnet.csv')
featCols=rawDF.columns[0:len(rawDF.columns)-1]
labelCol=rawDF.columns[-1]
featCols,labelCol
from pyspark.ml.feature import VectorAssembler,StandardScaler
vecAssembler=VectorAssembler(inputCols=featCols,outputCol='features')
vecDiabDF=vecAssembler.transform(rawDF)
stdScaler=StandardScaler(inputCol='features',outputCol='scaledFeatures',withMean=True,withStd=True)
scalerModel=stdScaler.fit(vecDiabDF)
scaledVecDiabDF=scalerModel.transform(vecDiabDF)
scaledVecDiabDF=scaledVecDiabDF.select(['scaledFeatures',labelCol])
scaledVecDiabDF.show(3,truncate=True)
train,test=scaledVecDiabDF.randomSplit([0.8,0.2])
train.count(),test.count()
from pyspark.ml.classification import LogisticRegression
lReg=LogisticRegression(featuresCol='scaledFeatures',labelCol='diabetes',maxIter=1000,elasticNetParam=0.01)
lr_model=lReg.fit(train)
lr_model.coefficients,lr_model.intercept
training_summary=lr_model.summary
print('Acc:',training_summary.accuracy*100,'%')
test_model=lr_model.evaluate(test)
print('Acc on test_data:',test_model.accuracy*100,'%')