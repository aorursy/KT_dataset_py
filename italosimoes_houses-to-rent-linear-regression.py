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



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv('/kaggle/input/brasilian-houses-to-rent/houses_to_rent_v2.csv')
df.head()
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
df['animal'].replace('acept',1,inplace=True)
df['animal'].replace('not acept',0,inplace=True)
df['furniture'].replace('furnished',1,inplace=True)
df['furniture'].replace('not furnished',0,inplace=True)
df.head()
df.describe()
plt.figure(figsize=(10,6))

sns.distplot(df['rent amount (R$)'])
sns.countplot(df['animal'],data=df)
sns.barplot(x='animal',y='rent amount (R$)',data=df)
sns.barplot(x='animal',y='property tax (R$)',data=df)
sns.barplot(x='furniture',y='rent amount (R$)',data=df)
plt.figure(figsize=(15,6))

sns.heatmap(df.corr(),annot=True)
plt.figure(figsize=(10,6))

sns.countplot(x='animal', hue='city', data=df)
plt.figure(figsize=(10,6))

sns.countplot(x='furniture', hue='city', data=df)
X = df[['area','rooms','bathroom','parking spaces','animal','furniture','hoa (R$)',

        'fire insurance (R$)']]
y = df['rent amount (R$)']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=101)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
lm.coef_
lm.intercept_
predict = lm.predict(X_test)
plt.figure(figsize=(10,6))

plt.scatter(y_test,predict)

plt.xlabel('Y teste')

plt.ylabel('valores previstos')
from sklearn import metrics
print('Mae',metrics.mean_absolute_error(y_test,predict))

print('Mse',metrics.mean_squared_error(y_test,predict))

print('Rmse',np.sqrt(metrics.mean_squared_error(y_test,predict)))