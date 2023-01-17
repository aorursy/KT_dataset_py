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
import pandas as pd

df=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
df.info()

print(df.head())
df.drop(df.select_dtypes(include='O'),axis=1,inplace=True)
df.head()
import seaborn as sns

sns.heatmap(df.isnull())
for i in df.columns[df.isnull().any()]:

    df[i].fillna(df[i].mean(),inplace=True)
sns.heatmap(df.isnull())
sns.heatmap(df.corr())
sns.distplot(df['SalePrice'])
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(df.drop('SalePrice',axis=1),df['SalePrice'],test_size=0.3)
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X_train,y_train)
predictions = lr.predict(X_test)

import matplotlib.pyplot as plt

plt.scatter(y_test,predictions)
sns.distplot((y_test-predictions),bins=50)
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))

print('MSE:', metrics.mean_squared_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))