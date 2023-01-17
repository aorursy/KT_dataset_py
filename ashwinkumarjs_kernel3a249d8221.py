

import pandas as pd

import numpy as np

from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import AdaBoostRegressor

from sklearn.model_selection import ShuffleSplit

import xgboost as xgb

import pylab as pl

import pickle

import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))
Wther2009=pd.read_excel('../input/retail-case-study-b11/WeatherData.xlsx','2009')

Wther2010=pd.read_excel('../input/retail-case-study-b11/WeatherData.xlsx','2010')

Wther2011=pd.read_excel('../input/retail-case-study-b11/WeatherData.xlsx','2011')

Wther2012=pd.read_excel('../input/retail-case-study-b11/WeatherData.xlsx','2012')

Wther2013=pd.read_excel('../input/retail-case-study-b11/WeatherData.xlsx','2013')

Wther2014=pd.read_excel('../input/retail-case-study-b11/WeatherData.xlsx','2014')

Wther2015=pd.read_excel('../input/retail-case-study-b11/WeatherData.xlsx','2015')

Wther2016=pd.read_excel('../input/retail-case-study-b11/WeatherData.xlsx','2016')
Wther2011[:10]


Wther=pd.concat([Wther2009,Wther2010,Wther2011,Wther2012,Wther2013,Wther2014,Wther2015,Wther2016],ignore_index=True)
Wther=Wther.drop(['Temp high (°C)','Temp low (°C)','Dew Point high (°C)','Dew Point low (°C)','Humidity\xa0(%) high','Humidity\xa0(%) low','Sea Level Press.\xa0(hPa) high','Sea Level Press.\xa0(hPa) low','Visibility\xa0(km) high','Visibility\xa0(km) low','Wind\xa0(km/h) high','Wind\xa0(km/h) low','WeatherEvent','Precip.\xa0(mm) sum'],axis=1)
Wther[:-10]


year={2009:1,2010:2,2011:3,2012:4,2013:5,2014:6}

month={'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6, 'Jul':7, 'Aug':8, 'Sep':9,

       'Oct':10, 'Nov':11, 'Dec':12}

Wther["Year"] = Wther["Year"].map(year)

Wther["Month"] = Wther["Month"].map(month)


list(Wther.columns)
Wther[:-10]
Wther_avg=Wther.groupby(['Temp avg (°C)','Dew Point avg (°C)','Humidity\xa0(%) avg','Sea Level Press.\xa0(hPa) avg','Visibility\xa0(km) avg','Wind\xa0(km/h) avg']).mean()
Wther_avg=Wther.groupby(['Year','Month'])['Temp avg (°C)','Dew Point avg (°C)','Humidity\xa0(%) avg','Sea Level Press.\xa0(hPa) avg','Visibility\xa0(km) avg','Wind\xa0(km/h) avg'].mean()
Wther_avg.shape
Wther_avg1=Wther_avg.values[:,:]
Wther_avg1[:10]
Wther_fin=pd.DataFrame(Wther_avg,columns=['Temp_avg','Dew_Point_avg','Humidity_avg','Sea_Level_Press_avg','Visibility_avg','Wind_avg'])
Wther_fin[:10]


Wther_fin.shape
data=pd.read_csv('../input/retail-case-study-b11/Train_Kaggle.csv')
data_Women=data[data['ProductCategory']=='WomenClothing']
data_Women[:-50]
data_Women=data_Women.fillna(value=3293)
data_Women[:-50]
data_Other=data[data['ProductCategory']=='OtherClothing']
data_Other=data_Other.fillna(value=1107)


data_Men=data[data['ProductCategory']=='MenClothing']
data_Men=data_Men.fillna(value=674)
data_Men[:10]
fin_data=pd.concat([data_Men,data_Women,data_Other])
fin_data[:-50]
#fin_data

dept={'MenClothing':1,'WomenClothing':2,'OtherClothing':3}

fin_data["ProductCategory"] = fin_data["ProductCategory"].map(dept)
fin_data[:-50]
year={2009:1,2010:2,2011:3,2012:4,2013:5,2014:6}

fin_data["Year"] = fin_data["Year"].map(year)
fin_data[:-10]
train_fin=fin_data
train_fin[:-10]
train_fin_Men=train_fin[train_fin['ProductCategory']==1].reset_index(drop=True)

train_fin_Women=train_fin[train_fin['ProductCategory']==2].reset_index(drop=True)

train_fin_Others=train_fin[train_fin['ProductCategory']==3].reset_index(drop=True)
print(train_fin_Men[:10])

print(train_fin_Women[:10])

print(train_fin_Others[:10])
holi=pd.read_excel('../input/retail-case-study-b11/Events_HolidaysData.xlsx')
holi[:10]
holi_fin=holi.drop(['Year','MonthDate'],axis=1)
holi_fin[:10]

#holi_fin.shape


full_men=pd.concat([train_fin_Men,holi_fin,Wther],axis=1)

full_women=pd.concat([train_fin_Women,holi_fin,Wther],axis=1)

full_others=pd.concat([train_fin_Others,holi_fin,Wther],axis=1)
full_men[:10]
cs = full_men['Sales(In ThousandDollars)']

full_men.drop(labels=['Sales(In ThousandDollars)'], axis=1,inplace = True)

full_men.insert(14, 'Sales(In ThousandDollars)', cs)
cs = full_women['Sales(In ThousandDollars)']

full_women.drop(labels=['Sales(In ThousandDollars)'], axis=1,inplace = True)

full_women.insert(14, 'Sales(In ThousandDollars)', cs)
cs = full_others['Sales(In ThousandDollars)']

full_others.drop(labels=['Sales(In ThousandDollars)'], axis=1,inplace = True)

full_others.insert(14, 'Sales(In ThousandDollars)', cs)
cs[:10]
full_men.values[:,:10]
full_men.values[:,10:]
full_men[:-10]

#print(full_women.head())

#print(full_others.head())


full_men_X=full_men.values[:,:10]

full_men_Y=full_men.values[:,10:]

full_others_X=full_others.values[:,:10]

full_others_Y=full_others.values[:,10:]

full_women_X=full_women.values[:,:10]

full_women_Y=full_women.values[:,10:]

full_men_Y[:10]
full_men_X.shape,full_others_X.shape,full_women_X.shape