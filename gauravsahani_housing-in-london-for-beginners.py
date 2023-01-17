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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import sklearn
df=pd.read_csv("../input/housing-in-london/housing_in_london.csv")

df.head()
df.describe()
df.isnull().sum()
sns.heatmap(df.isnull(),yticklabels=False,cbar=False)
df.drop(['no_of_houses','date'],axis=1,inplace=True)

df.drop(['recycling_pct','life_satisfaction','median_salary','mean_salary'],axis=1,inplace=True)

sns.heatmap(df.isnull(),yticklabels=False,cbar=False)
df['no_of_crimes']=df['no_of_crimes'].fillna(df['no_of_crimes'].mean())

sns.heatmap(df.isnull(),yticklabels=False,cbar=False)
df['houses_sold']=df['houses_sold'].fillna(df['houses_sold'].mean())

sns.heatmap(df.isnull(),yticklabels=False,cbar=False)
df.info()
df.shape
df['code']=df.code.str.replace('E','').astype(float)

df.info()
df['area'] = pd.factorize(df.area)[0]

df['area'] = df['area'].astype("float")

df.info()
df.describe()
df.head()
X=df[['area','code','houses_sold','no_of_crimes','borough_flag']]

y=df[['average_price']]

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.15,random_state=1,)

from sklearn.tree import DecisionTreeRegressor

model=DecisionTreeRegressor(random_state=0,min_samples_split=3)

model.fit(X_train,y_train)
prediction=(model.predict(X_test).astype(int))

print("predictions:",prediction)
from sklearn.metrics import r2_score

r2_score(prediction,y_test)
