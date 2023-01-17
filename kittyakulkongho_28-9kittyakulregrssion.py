import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv('../input/iris/Iris.csv')
df.head()
df = pd.read_csv('../input/iris/Iris.csv')
cols=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
sns.pairplot(data=df[cols])
df = pd.read_csv('../input/iris/Iris.csv')
cols=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
sns.pairplot(df[cols],kind='reg')
df = pd.read_csv('../input/advertising.csv/Advertising.csv')
df.head()
df = pd.read_csv('../input/advertising.csv/Advertising.csv')
cols=['TV','radio','newspaper','sales']
sns.pairplot(df[cols],kind='reg')

df = pd.read_csv('../input/advertising.csv/Advertising.csv')
df.head()
x=df.TV.values.reshape(-1,1)
y=df.sales.values.reshape(-1,1)
sns.lmplot(x='TV',y='sales',data=df)
model=LinearRegression()
model.fit(x,y)
model.coef_,model.intercept_

df = pd.read_csv('../input/advertising.csv/Advertising.csv')
df.head()
x=df.TV.values.reshape(-1,1)
y=df.sales.values.reshape(-1,1)
sns.lmplot(x='TV',y='sales',data=df)
model=LinearRegression()
model.fit(x,y)
model.coef_,model.intercept_
x_input=[[300]]
y_predict=model.predict(x_input)
y_predict