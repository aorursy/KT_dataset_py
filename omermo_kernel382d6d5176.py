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
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('../input/automobile-dataset/Automobile_data.csv')
df.head()
df.info()
df.isnull().sum()
df=df.replace('?',np.nan)
df.isnull().sum()
df=df.drop('normalized-losses',axis=1)
df.dropna(inplace=True)
df.isnull().sum()
for i in df.columns:
    try:
        df[i] = df[i].astype('float')
    except:
        df[i] = df[i]
df.info()
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(),annot=True)
df.drop(['peak-rpm','compression-ratio','stroke','symboling','city-mpg','highway-mpg'],axis=1,inplace=True)
sns.distplot(df.price)
df['fuel-system'].unique()
df['fuel-system'] = df['fuel-system'].astype('category')
df['fuel-system'] = df['fuel-system'].cat.codes
df.info()
x = ['make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style',
       'drive-wheels', 'engine-location','engine-type', 'num-of-cylinders']
for j in x:
    df[j] = df[j].astype('category')
    df[j] = df[j].cat.codes
df.info()
plt.figure(figsize=(18,15))
sns.heatmap(data=df.corr(),annot=True)
from sklearn.model_selection import train_test_split

X = df[['make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style',
       'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width',
       'height', 'curb-weight', 'engine-type', 'num-of-cylinders',
       'engine-size', 'fuel-system', 'bore', 'horsepower']]
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train,y_train)
reg.coef_
predict = reg.predict(X_test)
predict
plt.scatter(y_test,predict)
sns.set_style('whitegrid')
sns.distplot(y_test-predict)
from sklearn import metrics
metrics.mean_absolute_error(y_test,predict)
