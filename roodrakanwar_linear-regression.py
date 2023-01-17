import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
data = pd.read_csv("../input/vehicle-dataset-from-cardekho/car data.csv")
data.head()
data.isnull().sum()
data.describe
sns.relplot( y = 'Selling_Price', x = 'Kms_Driven',  data = data)
sns.catplot(x='Owner', y ='Selling_Price', kind = 'violin',data= data)
sns.catplot(x='Transmission', y ='Selling_Price', kind = 'swarm',data= data)
sns.catplot(x='Seller_Type', y ='Selling_Price', kind = 'swarm',data= data,hue = 'Fuel_Type')
sns.catplot(x='Year', y ='Selling_Price', kind = 'swarm',data= data)
correlation = data.corr()
plt.subplots(figsize=(10,15))

sns.heatmap(correlation, annot = True)
sns.jointplot(x = 'Present_Price', y ='Selling_Price', data=data, color = 'Green')
sns.pairplot(data = data)
dummy1 = pd.get_dummies(data.Fuel_Type)

dummy2 = pd.get_dummies(data.Seller_Type)

dummy3 = pd.get_dummies(data.Transmission)
merge = pd.concat([data,dummy1,dummy2,dummy3], axis = 'columns')
final = merge.drop(['Car_Name','Fuel_Type','Seller_Type','Transmission','CNG','Individual','Automatic','Owner','Kms_Driven'], axis = 'columns')
final

X = final.drop(['Selling_Price'],axis = 'columns')
y = final['Selling_Price']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20,random_state = 20)
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train,y_train)
model.score(X_test,y_test)