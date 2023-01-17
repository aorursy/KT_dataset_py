import os

import pandas as pd

import numpy as np





dataset=pd.read_csv("../input/kc-housesales-data/kc_house_data.csv")

data=dataset.copy()





data.info()



data.describe()



data.info()
dell=["id","date"]

data=data.drop(dell,axis=1)



'''data.isnull().sum()

sum(data["sqft_living"]<1000)

sum(data["sqft_living"]>10000)

sum(data["sqft_lot"]<2000)

sum(data["sqft_lot"]>50000)

sum(data["sqft_above"]<500)

sum(data["sqft_above"]>4000)

sum(data["sqft_basement"]>2000)

sum(data["sqft_living15"]<900)

sum(data["sqft_living15"]>3500)

data=data[(data["sqft_living"]>=1000) & (data["sqft_living"]<=10000) & (data["sqft_lot"]>=2000) & (data["sqft_lot"]<=50000) & (data["sqft_above"]>=500) & (data["sqft_above"]<=4000)]

data=data[(data["sqft_basement"]<=2000) & (data["sqft_living15"]<=3500) & (data["sqft_living15"]>=900)]

'''
data["yr_built"].describe()

data["age"]=2016-data["yr_built"]

data["age"].describe()

#data=data[(data["age"]>=10)]

data=data.drop("yr_built",axis=1)

data1=data.copy()

for i in range(0,21597):

  if(data1["yr_renovated"][i]>0):

    data1["yr_renovated"][i]=1
colmn=list(data1.columns)



feature= list(set(colmn)-set(["price"])) 



y1=data1["price"].values 

x1=data1[feature].values



y1=np.log(y1)
from  sklearn.preprocessing import StandardScaler

sn= StandardScaler();

x1=sn.fit_transform(x1)
from sklearn.model_selection import train_test_split,cross_val_score

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error
train_x,test_x,train_y,test_y= train_test_split(x1,y1,test_size=0.3,random_state=0)



lr=LinearRegression()

model=lr.fit(train_x,train_y)



print("Score")

print(lr.score(test_x, test_y)) 



r2test=model.score(test_x,test_y)

r2train=model.score(train_x,train_y)

print(r2train,r2test)

prediction=lr.predict(test_x)
import matplotlib.pyplot as plt

plt.scatter(prediction, test_y, color = 'red')

plt.plot(prediction, test_y, color = 'blue')

plt.show()

from sklearn.ensemble import RandomForestRegressor

rf=RandomForestRegressor(n_estimators=100,max_features="auto",max_depth=100,min_samples_leaf=4,min_samples_split=10,random_state=1)

model1=rf.fit(train_x,train_y)

print("Score")

print(rf.score(test_x, test_y)) 

prediction_rf=rf.predict(test_x)

r2test1=model1.score(test_x,test_y)

r2train1=model1.score(train_x,train_y)

print(r2train1,r2test1)

plt.scatter(prediction_rf, test_y, color = 'red')

plt.plot(prediction_rf, test_y, color = 'blue')

plt.show()