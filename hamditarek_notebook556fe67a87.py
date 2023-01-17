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
df = pd.read_csv('/kaggle/input/airquality/AirQualityUCI.csv')

df
df['Date'] = pd.to_datetime(df['Date'])

h = df['Time'].tolist()

l=[]

for i in range(len(h)):

    l.append(h[i].split(".")[0])

df['hour']=l

    
df['year'] = df['Date'].dt.year

df['month'] = df['Date'].dt.month

df['day'] = df['Date'].dt.day
df
df = df.drop('Date', axis=1)

df = df.drop('Time', axis=1)
df = df.fillna(df.mean())
X = df.drop('CO(GT)', axis=1).head(9356)

X_test = df.drop('CO(GT)', axis=1).tail(1)



y = df['CO(GT)'].values[:-1]
from sklearn.linear_model import LinearRegression



model = LinearRegression()
model.fit(X, y)
r_sq = model.score(X, y)

print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)
y_pred = model.predict(X_test)

print('predicted response:', y_pred, sep='\n')
import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

#Using Pearson Correlation

plt.figure(figsize=(12,10))

cor = df.corr()

sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)

plt.show()