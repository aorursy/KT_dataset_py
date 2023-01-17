# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/weather-dataset-rattle-package/weatherAUS.csv')
df.head()
df.info()
df.drop('RISK_MM',axis=1,inplace=True)
df['Date'] = pd.to_datetime(df['Date'])
df['Day'] = df['Date'].apply(lambda x: x.day)

df['Month'] = df['Date'].apply(lambda x: x.month)

df['Year'] = df['Date'].apply(lambda x: x.year)
df.drop('Date',axis=1,inplace=True)
df['RainTomorrow']
df['RainTomorrow'].unique()
df['RainTomorrow'] = df['RainTomorrow'].apply(lambda x: 1 if(x=='Yes') else 0)
df.info()
df['Location']
df['Location'].nunique()
plt.figure(figsize=(20,6))

sns.barplot(x='Location',y='RainTomorrow',data=df)
location = pd.get_dummies(df['Location'],drop_first=True)

df.drop('Location',axis=1,inplace=True)
df.select_dtypes(object).columns
df['WindGustDir']
df['WindGustDir'].nunique()
plt.figure(figsize=(12,6))

sns.barplot(x='WindGustDir',y='RainTomorrow',data=df)
df.groupby('WindGustDir')['RainTomorrow'].mean()
wgd = pd.get_dummies(df['WindGustDir'],drop_first=True)
df.drop('WindGustDir',axis=1,inplace=True)
df2 = pd.concat((location,wgd),axis=1)
df2.head()
df['WindDir9am']
df['WindDir9am'].nunique()
plt.figure(figsize=(12,6))

sns.barplot(x='WindDir9am',y='RainTomorrow',data=df)
df.groupby('WindDir9am')['RainTomorrow'].mean()
plt.figure(figsize=(12,6))

sns.countplot(x='WindDir9am',data=df)
df['WindDir9am'].value_counts()
wd9 = pd.get_dummies(df['WindDir9am'],drop_first=True)

df.drop('WindDir9am',axis=1,inplace=True)
df2 = pd.concat((df2,wd9),axis=1)
df.select_dtypes(object).columns
df['WindDir3pm']
df['WindDir3pm'].nunique()
plt.figure(figsize=(12,6))

sns.barplot(x='WindDir3pm',y='RainTomorrow',data=df)
plt.figure(figsize=(12,6))

sns.countplot(x='WindDir3pm',data=df)
wd3 = pd.get_dummies(df['WindDir3pm'],drop_first=True)

df.drop('WindDir3pm',axis=1,inplace=True)
df2 = pd.concat((df2,wd3),axis=1)
df['RainToday']
df['RainToday'].nunique()
sns.barplot(x='RainToday',y='RainTomorrow',data=df)
sns.countplot(x='RainToday',data=df)
df['RainToday'] = df['RainToday'].apply(lambda x: 1 if(x=='Yes') else 0)
df.select_dtypes(object).columns
from sklearn.linear_model import LinearRegression
df = pd.concat((df,df2),axis=1)
missingcols = list(df.isnull().sum().sort_values()[-16:].index.values)

mc = list(df.isnull().sum().sort_values()[-16:].index.values)
missingcols
def fill(df,col):

    X = df[df[col].isnull()==False].drop(missingcols,axis=1)

    y = df[col].dropna()

    na = df[df[col].isnull()].drop(missingcols,axis=1)

    

    lin = LinearRegression()

    lin.fit(X,y)

    pred = lin.predict(na)

    aux = 0

    

    for i in df[df[col].isnull()].index:

        df.at[i, col] = pred[aux]

        aux+=1

    missingcols.remove(col)

    return df
for i in mc:

    df = fill(df,i)
df.isnull().sum().sort_values(ascending=False)
df.info()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = df.drop('RainTomorrow',axis=1)

y = df['RainTomorrow']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
from sklearn.metrics import classification_report,confusion_matrix
from xgboost import XGBClassifier
xgbc = XGBClassifier()
xgbc.fit(X_train,y_train)
predictions = xgbc.predict(X_test)
print(classification_report(y_test,predictions))

print(confusion_matrix(y_test,predictions))