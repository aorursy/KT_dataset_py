import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
df=pd.read_csv('../input/car-price/datasets_794035_1363047_carprices.csv')
df
df=df.rename(columns={'Car Model':'Car_Model'})
df
##Now Remove Orginal Car_Model Column by passing Df
##It is Another Way
df=pd.get_dummies(df,columns=['Car_Model'])

df
df.info()
df.shape
lg=LinearRegression()
lg
df=df.rename(columns={'Sell Price':'Sell_Price'})

#train
lg.fit(df[['Car_Model_Audi','Mileage','Age']],df.Sell_Price)
lg.score(df[['Car_Model_Audi','Mileage','Age']],df.Sell_Price)
lg.predict([[3,69000,6]]).round(2)
df
dt=DecisionTreeClassifier()
##Decission Tree
dt
dt.fit(df[['Car_Model_Audi','Mileage','Age']],df.Sell_Price)
dt.score(df[['Car_Model_Audi','Mileage','Age']],df.Sell_Price)
##Accuracy
dt.predict([[3,69000,6]])
##We Will take this Price bcz its accuracy is 100%