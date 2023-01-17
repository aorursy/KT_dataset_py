# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

#-*-coding:utf-8 -*-

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from matplotlib import pyplot as plt

%matplotlib notebook

from sklearn.ensemble import RandomForestRegressor



plt.rcParams['font.sans-serif'] = ['KaiTi'] # 指定默认字体

plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显

tr_data = pd.read_csv("/kaggle/input/titanic/train.csv")

tr_data
tr_out = tr_data.loc[:,"Survived"]

te_data = pd.read_csv("/kaggle/input/titanic/test.csv")

te_pas = te_data["PassengerId"]

import re



def findNameFeature(str):

    if "Mr." in str :

        return "Mr"

    if "Mrs." in str:

        return "Mrs"

    if "Miss." in str:

        return "Miss"

    if "Master." in str:

        return "Master"

    return "other"

def processDate(tr_data):

    tr_data = pd.merge(tr_data,pd.get_dummies(tr_data["Pclass"],prefix="Pclass"),left_index=True,right_index=True)

    name_f = tr_data["Name"].apply(findNameFeature)

    tr_data = pd.merge(tr_data,pd.get_dummies(name_f),left_index=True,right_index=True)

    tr_data = tr_data.iloc[:,2:]

    tr_data = pd.merge(tr_data,pd.get_dummies(tr_data["Sex"],prefix="Sex"),left_index=True,right_index=True)

    tr_data = tr_data.iloc[:,1:]

    atr_data = tr_data.drop(["Ticket","Fare","Cabin","Embarked"],axis=1)

    ate_data = atr_data[atr_data.Age.isnull()].values

    atr_data = atr_data[atr_data['Age'].notnull()].values

    rfr = RandomForestRegressor(n_estimators=2000,random_state=0,n_jobs=-1)

    rfr.fit(atr_data[:,1:],atr_data[:,0])

    tr_data.loc[tr_data['Age'].isnull(),'Age'] = rfr.predict(ate_data[:,1:])

    tr_data['Agegroup'] = tr_data['Age'].apply(lambda x: 1 if x<18 else (2 if x<40 else (3 if x<60 else 4)))

    Agegroup = pd.get_dummies(tr_data['Agegroup'],prefix="Age")

    tr_data=pd.merge(tr_data.iloc[:,1:],Agegroup,left_index=True,right_index=True)

    tr_data = tr_data.drop(["SibSp","Parch","Ticket","Fare","Cabin","Agegroup"],axis=1)

    tr_data = pd.merge(tr_data.iloc[:,1:],pd.get_dummies(tr_data["Embarked"],prefix="Emb"),left_index=True,right_index=True)

    return tr_data

tr_data = processDate(tr_data.iloc[:,2:])

te_data = processDate(te_data.iloc[:,1:])

tr_data
te_data
from sklearn import linear_model

lr = linear_model.LogisticRegression(C=1.0,penalty='l2',tol=1e-6)

lr.fit(tr_data.values,tr_out.values)
predicts = lr.predict(te_data.values)

result = pd.DataFrame({"PassengerId":te_pas.values,"Survived":predicts.astype(np.int32)})
abs(lr.predict(tr_data.values)-tr_out.values).sum()/tr_out.count()
result
result.to_csv("result.csv",index=False)