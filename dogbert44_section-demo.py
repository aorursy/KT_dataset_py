# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

import seaborn as sns

import xgboost as xgb

from matplotlib import pyplot as plt



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#download dataset from kaggle and adjust path below

train = pd.read_csv('../input/cscie82-2020/train_data.csv')

test = pd.read_csv('../input/cscie82-2020/test_data.csv')

train.output.value_counts() #normalize=True
f, ax = plt.subplots(figsize=(10, 6))

sns.countplot(x="subject", hue="output", data=train);
pd.Series(train[train.subject=='K']['output']).value_counts() 

#Yes K is all 1's
sns.catplot(x="subject", hue="output", col="phase",data=train, kind="count", col_wrap=2 , height=3, aspect=1.5);

Y = train.output

train.drop('output',inplace=True, axis=1)
train_1 = train.drop(['state','subject','phase'], axis=1)

test_1 = test.drop(['state','subject','phase'], axis=1)
#Create a model on complete training set 

#Leaving CV for your HW

model = xgb.XGBClassifier(seed=82, n_estimators=1100 , max_depth=3, colsample_bylevel=0.8,

                          colsample_bytree=0.7,learning_rate=0.01, reg_lambda=0.1 , 

                           scale_pos_weight = 0.18357862) #missing = -999 #711/3873



model.fit(train_1,Y, eval_metric='auc')

preds = model.predict_proba(test_1) #predicting probabilities  #submit these predictions
submithis = pd.DataFrame([test_1.index,preds[:,1]]).T

submithis.columns = ['id','output']

submithis.id = submithis.id.astype(int)

submithis.to_csv('submission.csv',index=False)  #0.62095