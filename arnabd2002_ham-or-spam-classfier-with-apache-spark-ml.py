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
sc=SparkContext(appName='sentimentAnalysisApp')
sql=SQLContext(sparkContext=sc)
rawDF=sql.read.format('csv').options(header=True,inferSchema=True).load('../input/spam.csv')
rawDF.show(2)
filteredDF=rawDF.select(['v2','v1'])
filteredDF=filteredDF.dropna()
filteredDF.show(10)
(train,test)=filteredDF.randomSplit([0.8,0.2],seed=100)
train.count(),test.count()
from pyspark.ml.feature import Tokenizer,HashingTF,IDF,StringIndexer
from pyspark.sql.functions import shuffle
from pyspark.ml.pipeline import Pipeline
tokenizer=Tokenizer(inputCol='v2',outputCol='words')
hashingTF=HashingTF(numFeatures=10000,inputCol='words',outputCol='tf')
idf=IDF(inputCol='tf',outputCol='tfidf',minDocFreq=2)
label=StringIndexer(inputCol='v1',outputCol='label')
pipeline=Pipeline(stages=[tokenizer,hashingTF,idf,label])
pipelineModel=pipeline.fit(train)
train_df=pipelineModel.transform(train)
test_df=pipelineModel.transform(test)
from pyspark.ml.classification import LogisticRegression
lr=LogisticRegression(featuresCol='tfidf',labelCol='label')
lr_model=lr.fit(train_df)
model_eval=lr_model.summary
model_eval.accuracy
predDF=lr_model.transform(test_df)
predDF.select('v2','prediction').filter(predDF.prediction==1).show()