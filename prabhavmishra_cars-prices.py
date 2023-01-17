from sklearn.linear_model import LinearRegression

from tabulate import tabulate

import re

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

pd.options.mode.chained_assignment = None 

data=pd.read_csv('../input/used-cars-price-prediction/train-data.csv')

ar=data.values

hnames=data.columns
lb=LabelEncoder()

for i in range(2,8):

    if i==4 or i==3 :

        continue

    ar[:,i]=lb.fit_transform(ar[:,i])





data=pd.DataFrame(ar,columns=hnames)    

data['Mileage']=data['Mileage'].str.extract(r'(\d+\.\d+)',flags=0,expand=True).astype(float)

data['Engine']=data['Engine'].str.extract(r'(\d+)',expand=True).astype(float)

data['Power']=data['Power'].str.extract(r'(\d+\.\d+)',flags=0,expand=True).astype(float)

del data['Unnamed: 0']

data.Engine.fillna(data.Engine.mean(),inplace=True)

data['Mileage'].fillna(data['Mileage'].median(),inplace=True)

data['Price'].fillna(88.20,inplace=True)

data['Seats'].fillna(data['Seats'].mean(),inplace=True)

for i in range(data.shape[0]):

    if data.at[i,'Fuel_Type']==2:

        data.at[i,'Mileage']=20.532;

data.replace(data.loc[data['New_Price'].isnull(),['New_Price']],0,inplace=True)

replacement=data.loc[data['New_Price']!=0,['New_Price']]



replacement['New_Price']=replacement['New_Price'].str.strip().str[0:-5]

replacement['New_Price']=pd.to_numeric(replacement.New_Price)



data.loc[data['New_Price']!=0,['Price']]=replacement.values



del replacement

del data['New_Price']





#these lines will fill valees in those places in dataframe where value is 0

data.replace(data.loc[data['Mileage']==0,['Mileage']],data.Mileage.mean(),inplace=True)

data.at[3999,'Seats']=5

data.at[489,'Price']=87.76



'''Fowlling code block deals with the prediction of the missing power as power is a crucial feature in deciding the price.Missing Mileages are filled by mean as mostly lie in a predictable range around 20-ish'''

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import cross_val_score

x_test=data.loc[data.Power.isnull() & data.Engine.notnull()]

x_test.drop(['Name','Location','Power'],axis=1,inplace=True)

x_test.fillna(0,inplace=True)

x_train=data.loc[data.Power.notnull() & data.Engine.notnull()]

y_train=x_train['Power']

x_train.drop(['Name','Location','Power'],axis=1,inplace=True)

model=DecisionTreeRegressor(random_state=3)

model.fit(x_train,y_train)

y_pred=model.predict(x_test)

data.loc[data.Power.isnull() & data.Engine.notnull(),['Power']]=y_pred





print(data)

from sklearn.model_selection import train_test_split

X=data.drop(['Price','Name'],axis=1)

y=data['Price']

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=49)



from sklearn.ensemble import RandomForestRegressor





regressor1= RandomForestRegressor(n_estimators=369,random_state=19,bootstrap=True)

regressor1.fit(x_train,y_train)

regressor1.score(x_test,y_test)

from sklearn.tree import DecisionTreeRegressor

rgr2=DecisionTreeRegressor(criterion='mse',random_state=42,min_samples_split=10,max_features='log2')

rgr2.fit(x_train,y_train)

rgr2.score(x_test,y_test)
from sklearn.ensemble import AdaBoostRegressor

from sklearn.tree import DecisionTreeRegressor

rgrA=AdaBoostRegressor(DecisionTreeRegressor(max_depth=28),n_estimators=270,learning_rate=0.7)

rgrA.fit(x_train,y_train)

rgrA.score(x_test,y_test)

                      

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.tree import DecisionTreeRegressor

rgrG=GradientBoostingRegressor(max_depth=11,n_estimators=400,learning_rate=0.7)

rgrG.fit(x_train,y_train)

rgrG.score(x_test,y_test)