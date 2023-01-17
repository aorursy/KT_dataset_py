import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
house = pd.read_csv("../input/dataset/USA_Housing.csv")
house.head()
house.columns = ['Income', 'Age', 'No_rooms' , 'No_bedrooms' , 'Population' , 'Price' , 'Address']
house.head()
names = ['Income', 'Age', 'No_rooms' , 'No_bedrooms' , 'Population' , 'Price']
for i in names:
    house[i] = (house[i] - min(house[i])) / (max(house[i]) - min(house[i]))
house.head()
target = house['Price']
input = house.drop(['Price','Address'],axis = 1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(input, target, test_size=0.20, random_state=42)
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X_train, y_train)
Y_predict = reg.predict(X_test)
(abs(y_test-Y_predict)).describe()
reg.score(X_train, y_train)
# Same using mean normalisation

house = pd.read_csv("../input/dataset/USA_Housing.csv")
house.columns = ['Income', 'Age', 'No_rooms' , 'No_bedrooms' , 'Population' , 'Price' , 'Address']
names = ['Income', 'Age', 'No_rooms' , 'No_bedrooms' , 'Population' , 'Price']
for i in names:
    house[i] = (house[i] - house[i].mean()) / (max(house[i]) - min(house[i]))
house.head()
target = house['Price']
input = house.drop(['Price','Address'],axis = 1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(input, target, test_size=0.05, random_state=42)
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X_train, y_train)
Y_predict = reg.predict(X_test)
(abs(y_test-Y_predict)).describe()
reg.score(X_train, y_train)
house.corr()
