import numpy as np 
import pandas as pd 
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv('../input/videogamesales/vgsales.csv')
df.head()
df = pd.read_csv('../input/videogamesales/vgsales.csv')
print(df[101:105].NA_Sales)
df = pd.read_csv('../input/videogamesales/vgsales.csv')
df.head()
cols =['NA_Sales','EU_Sales']
sns.pairplot(df,kind='reg')
df = pd.read_csv('../input/videogamesales/vgsales.csv')
df.head()
cols =['NA_Sales','EU_Sales','JP_Sales','Global_Sales']
x=df.Global_Sales.values.reshape(-1,1)
y=df.EU_Sales.values.reshape(-1,1)
sns.lmplot(x='Global_Sales',y='EU_Sales',data=df)
model=LinearRegression()
model.fit(x,y)
model.coef_,model.intercept_
y=(0.29340257*X)+(-0.01103446)
df = pd.read_csv('../input/videogamesales/vgsales.csv')
df.head()
cols =['NA_Sales','EU_Sales','JP_Sales','Global_Sales']
x=df.Global_Sales.values.reshape(-1,1)
y=df.EU_Sales.values.reshape(-1,1)
sns.lmplot(x='Global_Sales',y='EU_Sales',data=df)
model=LinearRegression()
model.fit(x,y)
model.coef_,model.intercept_
x_input=[[600]]
y_predict=model.predict(x_input)
y_predict