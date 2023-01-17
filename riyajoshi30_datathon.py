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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
df2 = pd.read_csv('solartest.csv')
df = pd.read_csv('solartrain.csv')
df.head()
df['Temperature'] = df['Temperature'].fillna(df['Temperature'].median()) 
df['Speed'] = df['Speed'].fillna( df['Speed'].median()) 
df['Pressure'] = df['Pressure'].fillna( df['Pressure'].mean())
df['Humidity'] = df['Humidity'].fillna( df['Humidity'].median())
df['WindDirection(Degrees)'] = df['WindDirection(Degrees)'].fillna( df['WindDirection(Degrees)'].median()) 
df['Radiation'] = df['Radiation'].fillna( df['Radiation'].mean())
df['Time'] = df['Time'].fillna(method='ffill')

df.drop(['Data', 'TimeSunRise', 'TimeSunSet'],axis=1, inplace=True)
for column in df[['UNIXTime']]:
    mode = df[column].mode()
    df[column] = df[column].fillna(mode)
df.columns[df.isna().any()]
df.head()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
x,y =  df[['UNIXTime','Temperature','Pressure',
       'Speed','Humidity','Time']], df['Radiation']
x_train, x_test, y_train , y_test = train_test_split(x,y,test_size = 0.23, random_state = 1)
model = LinearRegression()
model.fit(x_train,y_train)
predictions = model.predict(x_test)
print('Accuracy of first model : ' ,round(r2_score(y_test, predictions)*100,2))
X = df[['UNIXTime','Temperature', 'Pressure', 'Speed','Humidity']]
y=df['Radiation']
from sklearn.preprocessing import PolynomialFeatures
x1 = PolynomialFeatures(degree=2, include_bias=False).fit_transform(x)
model = LinearRegression().fit(x1, y)
fin = model.score(x1,y)
fin
df2['Temperature'] = df2['Temperature'].fillna(df2['Temperature'].median()) 
df2['Speed'] = df2['Speed'].fillna( df2['Speed'].median()) 
df2['Pressure'] = df2['Pressure'].fillna( df2['Pressure'].mean())
df2['Humidity'] = df2['Humidity'].fillna( df2['Humidity'].median())
df2['WindDirection(Degrees)'] = df2['WindDirection(Degrees)'].fillna( df2['WindDirection(Degrees)'].median())
df2.drop(['Data', 'TimeSunRise', 'TimeSunSet', 'Time'],axis=1, inplace=True)
df2['UNIXTime'] = df2['UNIXTime'].fillna( df2['UNIXTime'].mean())

df2.columns[df2.isna().any()]

X_test = df2[['UNIXTime','WindDirection(Degrees)', 'Temperature', 'Pressure', 'Speed','Humidity']]
x2 =PolynomialFeatures(degree=2, include_bias=False).fit_transform(X_test)
y_pred = model.predict(x2)
df2['Radiation'] = y_pred
df2.apply(lambda col: col.drop_duplicates().reset_index(drop=True))
df2
dff=df2[['ID','Radiation']]
dff
dff.to_csv("submissionfinal10.csv", index= False)
