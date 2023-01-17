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
#Let's first read the data and store it in df
df=pd.read_csv('/kaggle/input/car-price-prediction/CarPrice_Assignment.csv')
df.head(10)# Showing the first 10 rows of df or dataframe
#Showing the last 10 rows of dataframe
df.tail(10)
#Showing the total no of rows,mean, standard deviation,minimum,percentiles and maximum values
df.describe().T
df.describe()
#Checking whether there is any null values
df.isnull().sum()#0 indicates no null values
plt.figure(figsize=(16,9))
sns.heatmap(df.isnull())
#Printing the data type of each feature
df.info()
df.columns.values
df=df.drop(['CarName','car_ID'],axis=1)
df.head(5)
feature=['fueltype','symboling','wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight', 'enginesize',
       'boreratio', 'stroke', 'compressionratio', 'horsepower', 'peakrpm',
       'citympg', 'highwaympg','price']

x=df[feature]
x

x['fueltype']=[1 if k=='gas' else 0 for k in x['fueltype']]

#Visualising the correlation Heatmap by seaborn
plt.figure(figsize=(16,9))
ax=sns.heatmap(x.corr(),cmap='Dark2',annot=True, linewidths=4,
    linecolor='black')
ax.set(title="Correlation Heatmap")
x.corr()
# stroke,compressionratio and peakrpm has too low correlation with price so they are removed 
x=x.drop(['stroke','compressionratio','peakrpm'],axis=1)
x
#Spliting the independent features from the target feature
x=x.drop('price',axis=1)
y=df['price']#Target feature
x.shape
y.shape
x.info()
#Spliting the data for training and testing. 20% of the data is used for testing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=0)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
#Model creation
from sklearn.tree import DecisionTreeRegressor
l=DecisionTreeRegressor()
l.fit(x_train,y_train)
y_pred=l.predict(x_test)
print("Training Accuracy:- ",l.score(x_train,y_train)*100)
plt.title('Residual Plot',size=16)
sns.residplot(y_test,y_pred,color='r'
              )
plt.xlabel('Y_pred',size=12)
plt.ylabel('Residues',size=12)
from sklearn.metrics import mean_squared_error,r2_score
print("MSE:- ",mean_squared_error(y_test,y_pred))

print("R2_score:-",r2_score(y_test,y_pred))
from sklearn.linear_model import LinearRegression
l=LinearRegression()
l.fit(x_train,y_train)
y_pred=l.predict(x_test)
print("Training Accuracy:- ",l.score(x_train,y_train)*100)
plt.title('Residual Plot',size=16)
sns.residplot(y_test,y_pred,color='r'
              )
plt.xlabel('Y_pred',size=12)
plt.ylabel('Residues',size=12)
print("MSE:- ",mean_squared_error(y_test,y_pred))

print("R2_score:-",r2_score(y_test,y_pred))
from sklearn.svm import SVR
l=SVR(kernel='linear',epsilon=0.01)
l.fit(x_train,y_train)
y_pred=l.predict(x_test)
print("Training Accuracy:- ",l.score(x_train,y_train)*100)
plt.title('Residual Plot',size=16)
sns.residplot(y_test,y_pred,color='r'
              )
plt.xlabel('Y_pred',size=12)
plt.ylabel('Residues',size=12)
print("MSE:- ",mean_squared_error(y_test,y_pred))
print("R2_score:-",r2_score(y_test,y_pred))
from sklearn.ensemble import RandomForestRegressor
l=RandomForestRegressor(max_depth=2, random_state=0,criterion='mae',n_estimators=90,min_samples_leaf=3)
l.fit(x_train,y_train)
y_pred=l.predict(x_test)
print("Training Accuracy:- ",l.score(x_train,y_train)*100)
plt.title('Residual Plot',size=16)
sns.residplot(y_test,y_pred,color='r'
              )
plt.xlabel('Y_pred',size=12)
plt.ylabel('Residues',size=12)
print("MSE:- ",mean_squared_error(y_test,y_pred))
print("R2_score:-",r2_score(y_test,y_pred))