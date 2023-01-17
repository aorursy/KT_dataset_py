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
df = pd.read_csv('/kaggle/input/ccdata/CC GENERAL.csv')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings(action="ignore")
df.isnull().sum()
df.head()
df.CREDIT_LIMIT= df.CREDIT_LIMIT.fillna(df.CREDIT_LIMIT.median())
df.CREDIT_LIMIT.isnull().sum()
df.drop(columns=['CUST_ID'], inplace= True)
train = df[df.MINIMUM_PAYMENTS.isna() == False]
test = df[df.MINIMUM_PAYMENTS.isna()  ]
x_train = train.drop(columns=['MINIMUM_PAYMENTS'])
y_train = train.MINIMUM_PAYMENTS
x_test = test.drop(columns=['MINIMUM_PAYMENTS'])
y_test = test.MINIMUM_PAYMENTS
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)

len(y_pred), df.MINIMUM_PAYMENTS.isnull().sum()
df.MINIMUM_PAYMENTS= df.MINIMUM_PAYMENTS.fillna(pd.Series(y_pred, index = df[df['MINIMUM_PAYMENTS'].isnull()].index ))
df.isnull().sum()
df.describe()
df = pd.DataFrame(scaler.fit_transform(df))
from sklearn.cluster import KMeans
n_clusters=30
cost=[]
for i in range(1,n_clusters):
    kmean= KMeans(i)
    kmean.fit(df)
    cost.append(kmean.inertia_)  

plt.plot(cost)


