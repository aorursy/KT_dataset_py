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
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})
df = pd.read_csv('/kaggle/input/insurance/insurance.csv')
df.head()
df.info()
df.shape
df.isnull().sum()
df.head()

df['sex'].value_counts()
df['sex'].value_counts().plot(kind='bar')
df['region'].value_counts().plot(kind='pie')
df['smoker'].value_counts().plot(kind='bar')
df.groupby('smoker')['charges'].mean()
df.groupby('children')['charges'].mean()
sns.countplot(data=df,x='region',hue='smoker')
df['children'].value_counts().plot(kind='bar')

sns.countplot(x='sex',hue='smoker',data=df)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
df = pd.get_dummies(df,drop_first=True)
df.head()
cols=['age','bmi','children','sex_male','smoker_yes','region_northwest','region_southeast','region_southwest','charges']
df = df[cols]
df.head()
X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
y = y.reshape(-1,1)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)
print(X.shape,y.shape)
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.2)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print(model.score(X_test,y_test))

from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)
regressor.score(X_test,y_test)
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)
regressor.score(X_test,y_test)
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=200,random_state=0)
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)
regressor.score(X_test,y_test)