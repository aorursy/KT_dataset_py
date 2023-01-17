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

import plotly.express as ply

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

df=pd.read_csv('../input/used-car-dataset-ford-and-mercedes/bmw.csv')   

df.isnull().any().sum()
df.isna().any().sum()
df.info()
df['model'].unique()
df['transmission'].unique()
df['fuelType'].unique()
corr_=df.corr()

sns.heatmap(corr_,annot=True)
plt.figure(figsize=(17,5))

sns.countplot(df['model'])

plt.title('Model')
df_ser_=df.model.value_counts()

df_ser=pd.DataFrame(df_ser_)

labels=df['model'].unique()

sizes=df_ser['model']
fig1,ax1=plt.subplots()

ax1.pie(sizes,explode=None,labels=labels,autopct='%1.1f%%',shadow=True,startangle=90)

ax1.axis('equal')

plt.show()
sns.countplot(df['fuelType'])

plt.title('Fuel Type')
sns.countplot(df['transmission'])

plt.title('Transmission type')
sns.countplot(df['engineSize'])

plt.title('Engine Size')
plt.figure(figsize=(20,10))

plt.scatter(df['model'],df['price'])

plt.title('Model Vs Fuel type')

plt.show()
plt.figure(figsize=(50,30))

sns.jointplot(x='engineSize',y='mpg',data=df)

plt.xlabel('engineSize')

plt.ylabel('mpg')

plt.grid()

plt.show()
plt.figure(figsize=(10,10))

plt.scatter(df['fuelType'],df['mpg'])

plt.xlabel('Fuel Type')

plt.ylabel('MPG')

plt.title('Fuel type vs MPG')

plt.show()
plt.figure(figsize=(20,10))

plt.scatter(df['model'],df['mpg'])

plt.xlabel('Model')

plt.ylabel('MPG')

plt.title('Model vs MPG')

plt.show()
plt.figure(figsize=(20,10))

plt.scatter(df['fuelType'],df['transmission'])

plt.xlabel('Fuel ')

plt.ylabel('Transmission')

plt.title('Model vs Transmission')

plt.show()
plt.figure(figsize=(20,10))

plt.scatter(df['tax'],df['model'])

plt.xlabel('Tax')

plt.ylabel('Model')

plt.title('Model vs MPG')

plt.show()
from sklearn.preprocessing import LabelEncoder

LE=LabelEncoder()
df['model']=LE.fit_transform(df['model'])

df['fuelType']=LE.fit_transform(df['fuelType'])

df['transmission']=LE.fit_transform(df['transmission'])
df.info()
features=['model','year','transmission','mileage','fuelType','tax','mpg','engineSize']
X=df[features]

Y=df['price']
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=1)
from sklearn.linear_model import LinearRegression

LR=LinearRegression()

lr=LR.fit(X_train,Y_train)

pred=lr.predict(X_test)
from sklearn.metrics import r2_score,mean_squared_error
print('R-Squared: ',r2_score(Y_test,pred)*100)
vals=pd.DataFrame({'Predicted':pred,'Actual':Y_test})

vals
from sklearn.linear_model import Ridge,ElasticNet

ridge=Ridge(alpha=2,max_iter=1000)
ridge.fit(X_train,Y_train)
Ridge_predict=ridge.predict(X_test)
ridge.score(X_test,Y_test)*100 
EN=ElasticNet(alpha=1,l1_ratio=1.001,max_iter=1000)

EN.fit(X_train,Y_train)
EN_pred=EN.predict(X_test)
EN.score(X_test,Y_test)*100
from sklearn.ensemble import GradientBoostingRegressor
GB=GradientBoostingRegressor(random_state=0)

GB.fit(X_test,Y_test)
GB_pred=GB.predict(X_test)

GB_pred
print('Performance Score(GB): ',GB.score(X_test,Y_test)*100)
from xgboost import XGBRegressor

XGB=XGBRegressor()

XGB.fit(X_train,Y_train)
XGB_pred=XGB.predict(X_test)
print('performance score(XGB): ',XGB.score(X_test,Y_test)*100) 
#For XGB

values=pd.DataFrame({'Predicted':XGB_pred,'Actual':Y_test})

values
#For GB

values=pd.DataFrame({'Predicted':GB_pred,'Actual':Y_test})

values
print('Linear Regression accuracy score: ',r2_score(Y_test,pred)*100)

print('Ridge Regression accuracy score: ',ridge.score(X_test,Y_test)*100 )

print('Elastic_Net Regression accuracy score: ',EN.score(X_test,Y_test)*100)

print('Gradient_Boosting Regression accuracy score: ',GB.score(X_test,Y_test)*100)

print('XGB Regression accuracy score: ',XGB.score(X_test,Y_test)*100)