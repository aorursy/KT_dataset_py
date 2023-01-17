# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pip install pyspark
from pyspark import SparkContext, SparkConf

from pyspark.sql import functions as F

from pyspark.sql import Window, SparkSession

from pyspark.sql.functions import col

from pyspark.sql.types import DoubleType, StringType, ArrayType, LongType, FloatType, DateType

from sklearn.model_selection import train_test_split

import sys

import xgboost as xgb

import numpy as np

from sklearn.metrics import roc_curve, auc

from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt

from sklearn.preprocessing import label_binarize

import lightgbm as lgb
spark = (SparkSession.builder

                  .appName('Toxic Comment Classification')

                  .enableHiveSupport()

                  .config("spark.executor.memory", "10G")

                  .config("spark.driver.memory","5G")

                  .config("spark.executor.cores","7")

                  .config("spark.python.worker.memory","4G")

                  .config("spark.driver.maxResultSize","0")

                  .config("spark.sql.crossJoin.enabled", "true")

                  .config("spark.serializer","org.apache.spark.serializer.KryoSerializer")

                  .config("spark.default.parallelism","2")

                  .config("spark.kryoserializer.buffer.max.mb", "2047").getOrCreate()

        )
df = spark.read.csv('/kaggle/input/liverpool-ion-switching/train.csv', header=True)
df.show()
description = df.describe().collect()

description
signal_stddev = float(description[2]['signal'])

open_channels_stddev = float(description[2]['open_channels'])



signal_mean = float(description[1]['signal'])

open_channels_mean = float(description[1]['open_channels'])
df1 = df.filter(col('time') <= 50)

df2 = df.filter(col('time') > 50)
last_2 = Window().orderBy('time').rowsBetween(-2, -1)

last_4 = Window().orderBy('time').rowsBetween(-4, -1)

last_8 = Window().orderBy('time').rowsBetween(-8, -1)

last_16 = Window().orderBy('time').rowsBetween(-16, -1)

last_32 = Window().orderBy('time').rowsBetween(-32, -1)



lead_2 = Window().orderBy('time').rowsBetween(1, 2)

lead_4 = Window().orderBy('time').rowsBetween(1, 4)

lead_8 = Window().orderBy('time').rowsBetween(1, 8)

lead_16 = Window().orderBy('time').rowsBetween(1, 16)

lead_32 = Window().orderBy('time').rowsBetween(1, 32)
df1 = df1.withColumn('last_2_sig_mean', F.mean('signal').over(last_2))

df1 = df1.withColumn('last_4_sig_mean', F.mean('signal').over(last_4))

df1 = df1.withColumn('last_8_sig_mean', F.mean('signal').over(last_8))

df1 = df1.withColumn('last_16_sig_mean', F.mean('signal').over(last_16))

df1 = df1.withColumn('last_32_sig_mean', F.mean('signal').over(last_32))

df1 = df1.withColumn('last_64_sig_mean', F.mean('signal').over(last_64))

df1 = df1.withColumn('last_128_sig_mean', F.mean('signal').over(last_128))



df1 = df1.withColumn('lead_2_sig_mean', F.mean('signal').over(lead_2))

df1 = df1.withColumn('lead_4_sig_mean', F.mean('signal').over(lead_4))

df1 = df1.withColumn('lead_8_sig_mean', F.mean('signal').over(lead_8))

df1 = df1.withColumn('lead_16_sig_mean', F.mean('signal').over(lead_16))

df1 = df1.withColumn('lead_32_sig_mean', F.mean('signal').over(lead_32))

df1 = df1.withColumn('lead_64_sig_mean', F.mean('signal').over(lead_64))

df1 = df1.withColumn('lead_128_sig_mean', F.mean('signal').over(lead_128))



df2 = df2.withColumn('last_2_sig_mean', F.mean('signal').over(last_2))

df2 = df2.withColumn('last_4_sig_mean', F.mean('signal').over(last_4))

df2 = df2.withColumn('last_8_sig_mean', F.mean('signal').over(last_8))

df2 = df2.withColumn('last_16_sig_mean', F.mean('signal').over(last_16))

df2 = df2.withColumn('last_32_sig_mean', F.mean('signal').over(last_32))

df2 = df2.withColumn('last_64_sig_mean', F.mean('signal').over(last_64))

df2 = df2.withColumn('last_128_sig_mean', F.mean('signal').over(last_128))



df2 = df2.withColumn('lead_2_sig_mean', F.mean('signal').over(lead_2))

df2 = df2.withColumn('lead_4_sig_mean', F.mean('signal').over(lead_4))

df2 = df2.withColumn('lead_8_sig_mean', F.mean('signal').over(lead_8))

df2 = df2.withColumn('lead_16_sig_mean', F.mean('signal').over(lead_16))

df2 = df2.withColumn('lead_32_sig_mean', F.mean('signal').over(lead_32))

df2 = df2.withColumn('lead_64_sig_mean', F.mean('signal').over(lead_64))

df2 = df2.withColumn('lead_128_sig_mean', F.mean('signal').over(lead_128))





df1 = df1.toPandas()
df2 = df2.toPandas()
df = pd.concat([df1, df2], sort=False)
df['signal'] = df['signal'].astype(np.float64)
df = df.fillna(0)
df_train, df_test = train_test_split(df, test_size=0.33, random_state=42)
clf = xgb.XGBClassifier(

    n_estimators=5,

    max_depth=9,

    learning_rate=0.05,

    subsample=0.9,

    colsample_bytree=0.9,

    missing=-999,

    random_state=2000

)
all_but_open_channels = list(df.columns)

all_but_open_channels.remove('open_channels')

all_but_open_channels.remove('time')
%time clf.fit(df_train[all_but_open_channels], df_train['open_channels'])
probs = clf.predict_proba(df_test[all_but_open_channels])
df_test['open_channels']
y_test = label_binarize(df_test['open_channels'], classes=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])

y_score = probs



fpr = dict()

tpr = dict()

roc_auc = dict()

for i in range(11):

    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])

    roc_auc[i] = auc(fpr[i], tpr[i])



print(roc_auc)



# Compute micro-average ROC curve and ROC area

fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())

roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])



auc = roc_auc['micro']



print('AUC performance', auc)
auc
y_score.shape
df = spark.read.csv('/kaggle/input/liverpool-ion-switching/test.csv', header=True)
df = df.withColumn('last_2_sig_mean', F.mean('signal').over(last_2))

df = df.withColumn('last_4_sig_mean', F.mean('signal').over(last_4))

df = df.withColumn('last_8_sig_mean', F.mean('signal').over(last_8))

df = df.withColumn('last_16_sig_mean', F.mean('signal').over(last_16))

df = df.withColumn('history_mean', F.mean('signal').over(history))
df = df.toPandas()
df['signal'] = df['signal'].astype(np.float64)
df = df.fillna(0)
preds = clf.predict(df[all_but_open_channels])
submission = pd.read_csv('/kaggle/input/liverpool-ion-switching/sample_submission.csv')
submission
submission['open_channels'] = preds
submission['time'] = submission['time'].map(lambda x: '%.4f' % x)
submission.to_csv('submission.csv', index=False)
submission[submission['time'] == 500.0010]