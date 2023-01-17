# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.options.mode.chained_assignment = None  # default='warn'

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train=pd.read_csv('../input/train.csv')
train.head()
#relevent data
trainDf=train[['Pclass','Sex','Age','Fare','Survived','SibSp','Parch']].head(700)
trainDf.dropna(axis=0,inplace=True)
trainDf.fillna(100,inplace=True)
trainDf['SexCode']=trainDf['Sex']=='male'
#trainDf['Survived']=trainDf['Survived']==1
trainDf.head()
target=trainDf.Survived
target.head()
inputs=trainDf[['Pclass','SexCode','Age','Fare','SibSp','Parch']]
from sklearn.tree import DecisionTreeRegressor
TitanicModel=DecisionTreeRegressor()

TitanicModel.fit(inputs,target)
crossDf=train[['PassengerId','Pclass','Sex','Age','Fare','Survived','SibSp','Parch']].tail(100)
crossDf.head()
target_c=crossDf['Survived']
inputs_c=crossDf[['Pclass','Sex','Age','Fare','SibSp','Parch']]
inputs_c.fillna(value=100,inplace=True)
inputs_c['SexCode']=inputs_c['Sex']=='male'
inputs_c=inputs_c[['Pclass','SexCode','Age','Fare','SibSp','Parch']]
pred_c=TitanicModel.predict(inputs_c)
pred_c=pd.DataFrame(data=pred_c,
                 columns=['Survived'])

pred_c = pred_c.reset_index(drop=True)
dfTemp = crossDf.reset_index(drop=True)

pred_c['PassengerId']=dfTemp['PassengerId']
pred_c.loc[pred_c['Survived']>=0.5,'Survived']=1
pred_c.loc[pred_c['Survived']<0.5,'Survived']=0
pred_c.head(20)
from sklearn.metrics import mean_absolute_error
mean_absolute_error(dfTemp['Survived'],pred_c['Survived'])

validation=pred_c['Survived']==dfTemp['Survived']
validation.describe()
missmatch=dfTemp.loc[dfTemp['Survived']!=pred_c['Survived']]
missmatch.head(22)
testDf=pd.read_csv('../input/test.csv')
testDf.head()
testInp=testDf[['Pclass','Sex','Age','Fare','SibSp','Parch']]
testInp.fillna(value=100,inplace=True)
testInp['SexCode']=testInp['Sex']=='male'
testInp=testInp[['Pclass','SexCode','Age','Fare','SibSp','Parch']]
pred=TitanicModel.predict(testInp)
pred=pd.DataFrame(data=pred,
                 columns=['Survived'])
pred['PassengerId']=testDf['PassengerId']
pred.loc[pred['Survived']>=0.5,'Survived']=1
pred.loc[pred['Survived']<0.5,'Survived']=0
pred.head()
pred.to_csv('prediction.csv',index=False)
