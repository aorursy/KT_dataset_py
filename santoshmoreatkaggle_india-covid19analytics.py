import datetime

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

%matplotlib inline

import json 

import requests

from pandas.io.json import json_normalize 

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()





r = requests.get('https://api.covid19india.org/raw_data.json')

data=r.json()

dmp=json.dumps(data)

p_obj = json.loads(dmp)



df = json_normalize(data, 'raw_data', errors='ignore')

df['dateannounced'] = pd.to_datetime(df['dateannounced'],format='%d/%m/%Y', errors='coerce')

df['statuschangedate'] = pd.to_datetime(df['statuschangedate'],format='%d/%m/%Y', errors='coerce')



df['currentstatus']= df['currentstatus'].replace('Migrated', 3) 

df['currentstatus']= df['currentstatus'].replace('Recovered', 2) 

df['currentstatus']= df['currentstatus'].replace('Hospitalized', 1) 

df['currentstatus']= df['currentstatus'].replace('Deceased', 9) 





df['currentstatus'] = pd.to_numeric(df['currentstatus'], downcast="integer",errors='coerce')

df['currentstatus'] = df['currentstatus'].fillna(0)

import pandas as pd

import numpy as np

import scipy.stats as sp

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression as lm

import datetime

from sklearn.model_selection import train_test_split

from sklearn import metrics

import math

from sklearn.preprocessing import PolynomialFeatures

%matplotlib inline



df  = df[df.currentstatus==1]

df1 = df[df.detectedstate!="Maharashtra"]

df2 = df[df.detectedstate=="Maharashtra"]

df3 = df1.groupby('dateannounced').currentstatus.cumsum(axis=0)

df4 = df2.groupby('dateannounced').currentstatus.cumsum(axis=0)





df_mah = df[df.detectedstate=="Maharashtra"]

df_mah1 = df_mah.groupby('dateannounced').currentstatus.cumsum(axis=0)



df_mum = df_mah[df_mah.detecteddistrict=="Mumbai"]

df_mum1 = df_mum.groupby('dateannounced').currentstatus.cumsum(axis=0)

df_mum_new = df_mum.groupby('dateannounced').currentstatus.cumsum(axis=0)





df_pun = df_mah[df_mah.detecteddistrict=="Pune"]

df_pun1 = df_pun.groupby('dateannounced').currentstatus.cumsum(axis=0)

df_pun_new = df_pun.groupby('dateannounced').currentstatus.cumsum(axis=0)



#### India Total #####



y=np.array(df1['currentstatus'].dropna().values.cumsum(), dtype=int)

x=np.array(df1['dateannounced'],dtype=int)



x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,test_size=0.2)

x_train= x_train.reshape(-1, 1)

y_train= y_train.reshape(-1, 1)

x_test = x_test.reshape(-1, 1)

y_test = y_test.reshape(-1, 1)



poly_features = PolynomialFeatures(degree=3)

x_train_poly = poly_features.fit_transform(x_train)



poly_model = lm()

poly_model.fit(x_train_poly, y_train)





x_pred = np.linspace(x.min(), x.max()+600000000000000, 200)         

x_pred = x_pred.reshape(-1, 1)

x_pred =np.array(x_pred ,dtype='datetime64[ns]')

x =np.array(x ,dtype='datetime64[ns]')



y_pred = poly_model.predict(poly_features.fit_transform(x_pred))

y_pred  = [0 if i < 0 else i for i in y_pred]



#### India Total #####





#### Maharashtra Total #####

y_mah=np.array(df_mah['currentstatus'].dropna().values.cumsum(), dtype=int)

x_mah=np.array(df_mah['dateannounced'].dropna(),dtype=float)





x_mah_train,x_mah_test,y_mah_train,y_mah_test=train_test_split(x_mah,y_mah,train_size=0.8,test_size=0.2)



x_mah_train= x_mah_train.reshape(-1, 1)

y_mah_train= y_mah_train.reshape(-1, 1)

x_mah_test = x_mah_test.reshape(-1, 1)

y_mah_test = y_mah_test.reshape(-1, 1)





x_mah_test=np.array(x_mah_test,dtype='datetime64[ns]')





poly_features = PolynomialFeatures(degree=3)

x_mah_train_poly = poly_features.fit_transform(x_mah_train)



poly_model = lm()

poly_model.fit(x_mah_train_poly, y_mah_train)







x_mah_pred = np.linspace(x_mah.min(), x_mah.max()+600000000000000, 200)         

x_mah_pred = x_mah_pred.reshape(-1, 1)

x_mah_pred =np.array(x_mah_pred ,dtype='datetime64[ns]')

x_mah =np.array(x_mah ,dtype='datetime64[ns]')

y_mah_pred = poly_model.predict(poly_features.fit_transform(x_mah_pred))

y_mah_pred  = [0 if i < 0 else i for i in y_mah_pred]

###### Maharashtra Total #####







###### Mumbai Total #####



y_mum=np.array(df_mum['currentstatus'].dropna().values.cumsum(), dtype=int)

x_mum=np.array(df_mum['dateannounced'].dropna(),dtype=float)





x_mum_train,x_mum_test,y_mum_train,y_mum_test=train_test_split(x_mum,y_mum,train_size=0.8,test_size=0.2)



x_mum_train= x_mum_train.reshape(-1, 1)

y_mum_train= y_mum_train.reshape(-1, 1)

x_mum_test = x_mum_test.reshape(-1, 1)

y_mum_test = y_mum_test.reshape(-1, 1)





x_mum_test=np.array(x_mum_test,dtype='datetime64[ns]')





poly_features = PolynomialFeatures(degree=3)

x_mum_train_poly = poly_features.fit_transform(x_mum_train)



poly_model_mum= lm()

poly_model_mum.fit(x_mum_train_poly, y_mum_train)





x_mum_pred = np.linspace(x_mum.min(), x_mum.max()+600000000000000, 200)         

x_mum_pred = x_mum_pred.reshape(-1, 1)

x_mum_pred =np.array(x_mum_pred ,dtype='datetime64[ns]')

x_mum =np.array(x_mum ,dtype='datetime64[ns]')

y_mum_pred = poly_model_mum.predict(poly_features.fit_transform(x_mum_pred))

y_mum_pred  = [0 if i < 0 else i for i in y_mum_pred]





###### Mumbai Total #####



###### Pune Total #####





y_pun=np.array(df_pun['currentstatus'].dropna().values.cumsum(), dtype=int)

x_pun=np.array(df_pun['dateannounced'].dropna(),dtype=float)





x_pun_train,x_pun_test,y_pun_train,y_pun_test=train_test_split(x_pun,y_pun,train_size=0.8,test_size=0.2)



x_pun_train= x_pun_train.reshape(-1, 1)

y_pun_train= y_pun_train.reshape(-1, 1)

x_pun_test = x_pun_test.reshape(-1, 1)

y_pun_test = y_pun_test.reshape(-1, 1)





x_pun_test=np.array(x_pun_test,dtype='datetime64[ns]')





poly_features = PolynomialFeatures(degree=3)

x_pun_train_poly = poly_features.fit_transform(x_pun_train)



poly_model_pun= lm()

poly_model_pun.fit(x_pun_train_poly, y_pun_train)







x_pun_pred = np.linspace(x_pun.min(), x_pun.max()+600000000000000, 200)         

x_pun_pred = x_pun_pred.reshape(-1, 1)

x_pun_pred =np.array(x_pun_pred ,dtype='datetime64[ns]')

x_pun =np.array(x_pun ,dtype='datetime64[ns]')

y_pun_pred = poly_model_pun.predict(poly_features.fit_transform(x_pun_pred))

y_pun_pred  = [0 if i < 0 else i for i in y_pun_pred]







#### Pune Total ####







y1=np.array(df3.fillna(0),dtype=int)

y2=np.array(df4.fillna(0),dtype=int)

x1=np.array(df1['dateannounced'].dropna(),dtype=float)

x2=np.array(df2['dateannounced'].dropna(),dtype=float)





x1_train,x1_test,y1_train,y1_test=train_test_split(x1,y1,train_size=0.8,test_size=0.2)

x1_train= x1_train.reshape(-1, 1)

y1_train= y1_train.reshape(-1, 1)

x1_test = x1_test.reshape(-1, 1)

y1_test = y1_test.reshape(-1, 1)



x2_train,x2_test,y2_train,y2_test=train_test_split(x2,y2,train_size=0.8,test_size=0.2)

x2_train= x2_train.reshape(-1, 1)

y2_train= y2_train.reshape(-1, 1)

x2_test = x2_test.reshape(-1, 1)

y2_test = y2_test.reshape(-1, 1)



poly_features = PolynomialFeatures(degree=3)



x1_train_poly = poly_features.fit_transform(x1_train)

x2_train_poly = poly_features.fit_transform(x2_train)

  

poly_model1 = lm()

poly_model1.fit(x1_train_poly, y1_train)



poly_model2 = lm()

poly_model2.fit(x2_train_poly, y2_train)





x1_pred = np.linspace(x1.min(), x1.max()+600000000000000, 200)         

x1_pred = x1_pred.reshape(-1, 1)

x1_pred =np.array(x1_pred ,dtype='datetime64[ns]')

x1 =np.array(x1 ,dtype='datetime64[ns]')



x2_pred = np.linspace(x2.min(), x2.max()+600000000000000, 200)         

x2_pred = x2_pred.reshape(-1, 1)

x2_pred =np.array(x2_pred ,dtype='datetime64[ns]')

x2 =np.array(x2 ,dtype='datetime64[ns]')



y1_pred = poly_model1.predict(poly_features.fit_transform(x1_pred))

y1_pred  = [0 if i < 0 else i for i in y1_pred]



y2_pred = poly_model2.predict(poly_features.fit_transform(x2_pred))

y2_pred  = [0 if i < 0 else i for i in y2_pred]







#### Mumbai New #####

y_new_mum=np.array(df_mum_new.fillna(0),dtype=int)

x_new_mum=np.array(df_mum['dateannounced'].dropna(),dtype=float)





x_new_mum_train,x_new_mum_test,y_new_mum_train,y_new_mum_test=train_test_split(x_new_mum,y_new_mum,train_size=0.8,test_size=0.2)

x_new_mum_train= x_new_mum_train.reshape(-1, 1)

y_new_mum_train= y_new_mum_train.reshape(-1, 1)

x_new_mum_test = x_new_mum_test.reshape(-1, 1)

y_new_mum_test = y_new_mum_test.reshape(-1, 1)

poly_features = PolynomialFeatures(degree=3)



x_new_mum_train_poly = poly_features.fit_transform(x_new_mum_train)

  

poly_model_new_mum = lm()

poly_model_new_mum.fit(x_new_mum_train_poly, y_new_mum_train)





x_new_mum_pred = np.linspace(x_new_mum.min(), x_new_mum.max()+600000000000000, 200)         

x_new_mum_pred = x_new_mum_pred.reshape(-1, 1)

x_new_mum_pred =np.array(x_new_mum_pred ,dtype='datetime64[ns]')

x_new_mum =np.array(x_new_mum ,dtype='datetime64[ns]')



y_new_mum_pred = poly_model_new_mum.predict(poly_features.fit_transform(x_new_mum_pred))

y_new_mum_pred  = [0 if i < 0 else i for i in y_new_mum_pred]

#### Mumbai New #####



#### Pune New ####





y_new_pun=np.array(df_pun_new.fillna(0),dtype=int)

x_new_pun=np.array(df_pun['dateannounced'].dropna(),dtype=float)





x_new_pun_train,x_new_pun_test,y_new_pun_train,y_new_pun_test=train_test_split(x_new_pun,y_new_pun,train_size=0.8,test_size=0.2)

x_new_pun_train= x_new_pun_train.reshape(-1, 1)

y_new_pun_train= y_new_pun_train.reshape(-1, 1)

x_new_pun_test = x_new_pun_test.reshape(-1, 1)

y_new_pun_test = y_new_pun_test.reshape(-1, 1)

poly_features = PolynomialFeatures(degree=3)



x_new_pun_train_poly = poly_features.fit_transform(x_new_pun_train)

  

poly_model_new_pun = lm()

poly_model_new_pun.fit(x_new_pun_train_poly, y_new_pun_train)





x_new_pun_pred = np.linspace(x_new_pun.min(), x_new_pun.max()+600000000000000, 200)         

x_new_pun_pred = x_new_pun_pred.reshape(-1, 1)

x_new_pun_pred =np.array(x_new_pun_pred ,dtype='datetime64[ns]')

x_new_pun =np.array(x_new_pun ,dtype='datetime64[ns]')



y_new_pun_pred = poly_model_new_pun.predict(poly_features.fit_transform(x_new_pun_pred))

y_new_pun_pred  = [0 if i < 0 else i for i in y_new_pun_pred]



#### Pune New ####







fig, (ax, ax_mah, ax_mum, ax_pun) = plt.subplots(1,4,figsize=(30,5),sharey=True)

fig, (ax1, ax2, ax_new_mum, ax_new_pun) = plt.subplots(1,4,figsize=(30,5),sharey=True)



ax.scatter(x_pred, y_pred, color='grey')

ax.scatter(x,y, edgecolor='Blue', facecolor='Blue', alpha=0.7)

ax.set_ylabel('India Total', fontsize=20)

ax.tick_params(labelrotation=45)

ax.grid(b=True, which='major', color='#666666', linestyle='-')

ax.minorticks_on()

ax.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

ax.legend

ax.set_title("India Excluding Maharashtra Total", fontsize=20)

ax.set_xlabel('Date', fontsize=20)

ax.legend( ('Polynomial Regression','India Excluding Maharashtra Total'), loc='upper left', shadow=True)





ax_mah.scatter(x_mah_pred, y_mah_pred, color='grey' )

ax_mah.scatter(x_mah,y_mah, edgecolor='Orange', facecolor='Orange', alpha=0.7 )

ax_mah.set_ylabel('Maharashtra Total', fontsize=20)

ax_mah.tick_params(labelrotation=45)

ax_mah.grid(b=True, which='major', color='#666666', linestyle='-')

ax_mah.minorticks_on()

ax_mah.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

ax_mah.legend

ax_mah.set_title("Maharashtra Total", fontsize=20)

ax_mah.set_xlabel('Date', fontsize=20)

ax_mah.legend( ('Polynomial Regression','Maharashtra Total'), loc='upper left', shadow=True)





ax_mum.scatter(x_mum_pred, y_mum_pred, color='grey' )

ax_mum.scatter(x_mum,y_mum, edgecolor='Orange', facecolor='Orange', alpha=0.7 )

ax_mum.set_ylabel('Mumbai Total', fontsize=20)

ax_mum.tick_params(labelrotation=45)

ax_mum.grid(b=True, which='major', color='#666666', linestyle='-')

ax_mum.minorticks_on()

ax_mum.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

ax_mum.legend

ax_mum.set_title("Mumbai Total",fontsize=20)

ax_mum.set_xlabel('Date', fontsize=20)

ax_mum.legend( ('Polynomial Regression','Mumbai Total'), loc='upper left', shadow=True)





ax_pun.scatter(x_pun_pred, y_pun_pred, color='grey' )

ax_pun.scatter(x_pun,y_pun, edgecolor='Orange', facecolor='Orange', alpha=0.7 )

ax_pun.set_ylabel('Pune Total', fontsize=20)

ax_pun.tick_params(labelrotation=45)

ax_pun.grid(b=True, which='major', color='#666666', linestyle='-')

ax_pun.minorticks_on()

ax_pun.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

ax_pun.legend

ax_pun.set_title("Pune Total",fontsize=20)

ax_pun.set_xlabel('Date', fontsize=20)

ax_pun.legend( ('Polynomial Regression','Pune Total'), loc='upper left', shadow=True)





ax1.scatter(x1_pred, y1_pred, color='grey' )

ax1.scatter(x1,y1, edgecolor='Blue', facecolor='Blue', alpha=0.7 )

ax1.set_ylabel('Daily New - India Exluding Maharashtra', fontsize=20)

ax1.tick_params(labelrotation=45)

ax1.grid(b=True, which='major', color='#666666', linestyle='-')

ax1.minorticks_on()

ax1.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

ax1.legend

ax1.set_title("India Excluding Maharashtra Daily New Cases")

ax1.set_xlabel('Date', fontsize=20)

ax1.set_xlabel('Date', fontsize=20)

ax1.legend( ('Polynomial Regression','India Exluding Maharashtra Daily New'), loc='upper left', shadow=True)



ax2.scatter(x2_pred, y2_pred, color='grey')

ax2.scatter(x2,y2, edgecolor='Orange', facecolor='Orange', alpha=0.7)

ax2.set_ylabel('Daily New - India  Maharashtra', fontsize=20)

ax2.set_title("Maharashtra Daily New Cases",fontsize=20)

ax2.set_xlabel('Date', fontsize=20)

ax2.tick_params(labelrotation=45)

ax2.grid(b=True, which='major', color='#666666', linestyle='-')

ax2.minorticks_on()

ax2.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

ax2.legend( ('Polynomial Regression','Maharashtra Daily New'), loc='upper left', shadow=True);





ax_new_mum.scatter(x_new_mum_pred, y_new_mum_pred, color='grey' )

ax_new_mum.scatter(x_new_mum,y_new_mum, edgecolor='Orange', facecolor='Orange', alpha=0.7 )

ax_new_mum.set_ylabel('Daily New - India  Maharashtra', fontsize=20)

ax_new_mum.tick_params(labelrotation=45)

ax_new_mum.grid(b=True, which='major', color='#666666', linestyle='-')

ax_new_mum.minorticks_on()

ax_new_mum.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

ax_new_mum.legend

ax_new_mum.set_title("Daily New Mumbai",fontsize=20)

ax_new_mum.set_xlabel('Date', fontsize=20)

ax_new_mum.legend( ('Polynomial Regression','Daily New Mumbai'), loc='upper left', shadow=True)





ax_new_pun.scatter(x_new_pun_pred, y_new_pun_pred, color='grey' )

ax_new_pun.scatter(x_new_pun,y_new_pun, edgecolor='Orange', facecolor='Orange', alpha=0.7 )

ax_new_pun.set_ylabel('Daily New - India  Maharashtra', fontsize=20)

ax_new_pun.tick_params(labelrotation=45)

ax_new_pun.grid(b=True, which='major', color='#666666', linestyle='-')

ax_new_pun.minorticks_on()

ax_new_pun.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

ax_new_pun.legend

ax_new_pun.set_title("Daily New Pune",fontsize=20)

ax_new_pun.set_xlabel('Date', fontsize=20)

ax_new_pun.legend( ('Polynomial Regression','IDaily New Pune'), loc='upper left', shadow=True);








