import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import plotly.express as px
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
data=pd.read_csv('../input/california-housing-prices/housing.csv')
data.head()
data.info()
data.describe()
sns.set_style('whitegrid')
sns.rugplot(data.isnull().sum())
data.isnull().sum()
data['total_bedrooms'].value_counts()
data.head()
sns.barplot(data=data,x='total_bedrooms',y='total_rooms',hue='ocean_proximity')
divide=data['total_bedrooms']/data['total_rooms']
divide
mean_of_bedroom_tototal_room=divide.mean()
mean_of_bedroom_tototal_room
data['total_bedrooms'].fillna(mean_of_bedroom_tototal_room*data['total_rooms'],inplace=True)
data.head()
sns.heatmap(data.isnull())
data.isnull().sum()
data.hist(bins=75,figsize=(16,14))
data.corr()
sns.heatmap(data.corr())
sns.jointplot(data=data,y='total_rooms',x='total_bedrooms',kind='reg')
sns.jointplot(data=data,y='total_rooms',x='total_bedrooms',kind='kde')
sns.pairplot(data)
data.head()
linreg=LinearRegression()
linreg
data.head(),data.shape
x=data.iloc[:,7:8].values
y=data.iloc[:,8].values
x.shape,y.shape
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=10)

linreg.fit(X_train,y_train)
y_pred=linreg.predict(X_test)
y_pred
linreg.predict([[9]])
linreg.score(X_test,y_test)
plt.plot(X_train,y_train,color='r')
plt.scatter(X_train,linreg.predict(X_train),color='g')
plt.xlabel('X_train')
plt.ylabel('y_train')
plt.title('X_train Vs y_train')
plt.show()
plt.plot(X_train[:10],y_train[:10],color='r')
plt.scatter(X_train[:10],linreg.predict(X_train)[:10],color='g')
plt.xlabel('X_train')
plt.ylabel('y_train')
plt.title('X_train Vs y_train')
plt.show()
plt.scatter(X_train[:10],y_train[:10],color='r')
plt.figure(figsize=(10,8))
plt.scatter(X_test, y_test,  color='#f57e42')
plt.plot(X_test, y_pred, color='black', linewidth=2)
plt.show()
plt.figure(figsize=(20,20))
fig=px.scatter(X_test, y_test,trendline='ols')
fig.show()
data.hist(figsize=(20,15))
