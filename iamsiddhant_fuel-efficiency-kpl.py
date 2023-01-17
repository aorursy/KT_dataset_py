#importing the necessary modules



import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
#reading the dataset



data=pd.read_csv('../input/autompg-dataset/auto-mpg.csv')
data.head()
#it is mentioned in the description of the dataset that some of the values in the horsepower are marked as  "?"



data[data['horsepower']=='?']
#filling the missing values in the horsepower column



data['horsepower'][32]=82

data['horsepower'][126]=70

data['horsepower'][330]=45

data['horsepower'][336]=86

data['horsepower'][354]=75

data['horsepower'][374]=82
#exracting the name of the brand 



list_brand=[]

for i in range(398):

    list_brand.append(data['car name'][i].split(" ")[0])
data['brand']=list_brand
# some of the names as written in short so converitng it a common name for each name



data['brand'] = data['brand'].replace(['volkswagen','vokswagen','vw'],'volkswagen')

data['brand'] = data['brand'].replace('maxda','mazda')

data['brand'] = data['brand'].replace('toyouta','toyota')

data['brand'] = data['brand'].replace('mercedes-benz','mercedes')

data['brand'] = data['brand'].replace('nissan','datsun')

data['brand'] = data['brand'].replace('capri','ford')

data['brand'] = data['brand'].replace(['chevroelt','chevy'],'chevrolet')
data['brand'].value_counts()
plt.figure(figsize=(8,6))

sns.countplot(y="brand",  data=data, palette="Greens_d",

              order=data.brand.value_counts().iloc[:15].index)
data.head()
#converting cu inches to cu cm(cc)



data['displacement_in_cc']=data['displacement']*16.387 
#converting lbs to kg



data['weight_in_kg']=data['weight']/2.205 
#converting mpg to kpl



data['mileage_kpl']=data['mpg']/2.352  
data.head()
data.info()
data['horsepower']=data['horsepower'].astype(int)
#origin and performance

plt.figure(figsize=(8,6))

sns.boxplot(x=data['origin'],y=data['mileage_kpl']);
plt.figure(figsize=(8,6))

sns.boxplot(y=data['mileage_kpl'],x=data['cylinders'])

plt.xlabel("no of cylinders");
#plotting pie chart for cylinders columns in the dataset



plt.figure(figsize=(14,8))

data.cylinders.value_counts().plot(kind='pie');
#plotting boxplots to observe the performance over years



plt.figure(figsize=(8,6))

sns.boxplot(x=data['model year'],y=data['mileage_kpl']);
#horsepower and mileage_kpl

plt.figure(figsize=(8,6))

plt.scatter(data.horsepower,data.mileage_kpl)

plt.xlabel('horsepower')

plt.ylabel('mileage in kpl');
#weight and mileage



plt.figure(figsize=(8,6))

plt.scatter(data.weight_in_kg,data.mileage_kpl)

plt.xlabel('weight in kg')

plt.ylabel('mileage in kpl');
X=data.drop(['displacement','weight','car name','brand','mpg'],axis=1)
X.head()
X.corr()
#plot heatmap



plt.figure(figsize=(10,6))

sns.heatmap(X.corr());
print(X.corr()["mileage_kpl"].sort_values(ascending=False))

X.head()
y=X['mileage_kpl']
X=X.drop(['mileage_kpl'],axis=1)
X.head()
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20)
lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)
lr.score(X_test,y_test)
# predicting the mileage of a vehicle with the trained model to check whether the results are satisfactory.



lr.predict([[4,120,9.7,90,3,1497,1160]])
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error 
model_rf = RandomForestRegressor(n_estimators=50)
model_rf.fit(X_train,y_train)
y_pred_rf = model_rf.predict(X_test)
from sklearn import metrics

import numpy as np

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_rf))

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_rf))

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_rf)))

model_rf.predict([[4,120,9.7,90,3,1497,1160]])