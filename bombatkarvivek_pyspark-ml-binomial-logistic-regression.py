! pip install pyspark
! curl  https://raw.githubusercontent.com/apache/spark/master/data/mllib/sample_libsvm_data.txt > sample_libsvm_data.txt
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("."))



# Any results you write to the current directory are saved as output.
! wc -l sample_libsvm_data.txt
from pyspark.sql import SparkSession

spark = SparkSession.builder.master("local[*]").getOrCreate()
from pyspark.ml.classification import LogisticRegression
training = spark.read.format("libsvm").load("sample_libsvm_data.txt")

pdf = training.toPandas()

pdf.T
lr = LogisticRegression(maxIter = 10, regParam = 0.3, elasticNetParam= 0.8 )
lr_model = lr.fit(training)
lr_model.coefficients
lr_model.intercept
mlr = LogisticRegression(maxIter = 10, regParam = 0.3, elasticNetParam=0.8, family="multinomial")
mlr_model = mlr.fit(training)
mlr_model.coefficientMatrix
mlr_model.interceptVector
train_summary = mlr_model.summary
obj_hist = train_summary.objectiveHistory

for obj in  obj_hist:

    print(obj)
train_summary.roc.show()
train_summary.areaUnderROC
f_measure = train_summary.fMeasureByThreshold

f_measure