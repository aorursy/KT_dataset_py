# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
x_train=pd.read_csv("/kaggle/input/titanic/train.csv")

x_test=pd.read_csv("/kaggle/input/titanic/test.csv")

x_train.head()
x_train.describe()
x_train.isna()
x_train.corr()
plt.matshow(x_train.corr())

plt.colorbar()
x_train.isna().sum()/x_train.shape[0]*100
columns_to_drop=['Name','Cabin','Ticket']
x_train.drop(columns_to_drop,axis=1,inplace=True)
from sklearn.preprocessing import OneHotEncoder

todummy_list=['Sex','Embarked']
def dummy_df(df,todummy_list):

    for x in todummy_list:

        dummies=pd.get_dummies(df[x],prefix=x,dummy_na=False)

        df=df.drop(x,axis=1)

        df=pd.concat([df,dummies],axis=1)

    return df
x_train=dummy_df(x_train,todummy_list)
x_train.dropna(inplace=True)

y_train=x_train.loc[:,"Survived"]

x_train.drop(['Survived'], axis=1,inplace=True)
x_train.isna().sum()
from sklearn.ensemble import RandomForestClassifier

clf=RandomForestClassifier(max_depth=10,random_state=0)

clf.fit(x_train,y_train)
x_test.drop(columns_to_drop,axis=1,inplace=True)

x_test=dummy_df(x_test,todummy_list)

x_test.fillna(x_test.mean(), inplace=True)
y_test=pd.concat([x_test.PassengerId,pd.Series(clf.predict(x_test),name='Survived')],axis=1)
y_test.to_csv('submission.csv',index=False)