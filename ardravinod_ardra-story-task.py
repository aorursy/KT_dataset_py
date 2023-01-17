# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#importing data-sets
df1 = pd.read_csv('../input/solar-power-generation-data/Plant_2_Generation_Data.csv')
df2 = pd.read_csv('../input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv')
#changing date-time into proper format
df1['DATE_TIME'] = pd.to_datetime(df1['DATE_TIME'],format = '%Y-%m-%d %H:%M:%S')
df1['DATE'] = df1['DATE_TIME'].apply(lambda x:x.date())
df1['TIME'] = df1['DATE_TIME'].apply(lambda x:x.time())
df1['DATE'] = pd.to_datetime(df1['DATE'],format = '%Y-%m-%d')
df2['DATE_TIME'] = pd.to_datetime(df2['DATE_TIME'],format = '%Y-%m-%d %H:%M:%S')
df2['DATE'] = df2['DATE_TIME'].apply(lambda x:x.date())
df2['TIME'] = df2['DATE_TIME'].apply(lambda x:x.time())
df2['DATE'] = pd.to_datetime(df2['DATE'],format = '%Y-%m-%d')
#generation data of plant 2
df1.info()
#weather sensor data of plant 2
df2.info()
df3 = pd.merge(df1, df2,on='DATE_TIME',how='left')
#merged data-set
df3.info()
import matplotlib.pyplot as plt
plt.figure(figsize=(12,8))
plt.plot(df3['IRRADIATION'],df3['MODULE_TEMPERATURE'],linestyle='',marker='o',alpha=0.5,c='r')
plt.xlabel('irradiation')
plt.ylabel('module temp')
plt.show()
plt.figure(figsize=(12,8))
plt.plot(df3['AC_POWER'],df3['DC_POWER'],linestyle='',marker='o',alpha=0.5,c='k')
plt.xlabel('ac power')
plt.ylabel('dc power')
plt.show()
df3.info()
#linear regression for irradiation and module temperature
#extracting two columns 
X = df3.iloc[:,13:14] #irradiation
y = df3.iloc[:,12] #module temperature
X.ndim
y.ndim
X.shape
y.shape
plt.scatter(X,y)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
X_train.shape
X_test.shape
y_train.shape
y_test.shape
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
y_pred = lin_reg.predict(X_test)
y_pred
y_test
plt.scatter(X_test,y_test,color='r',label='actual')
plt.scatter(X_test,y_pred,color='k',label='predicted')
plt.legend()
plt.show()
#slope
lin_reg.coef_
#y-intercept
lin_reg.intercept_
lin_reg.predict([[0.6]])
#using actual formulae (y=mx+c)
34.38535982*0.6+24.732291883106374
#choosing a random source key and plotting the graphs for date - time vs total yield and daily yield to find out at what point of time is the yield(efficiency of the plant) decreasing
df1['HOUR'] = pd.to_datetime(df1['TIME'],format='%H:%M:%S').dt.hour
df1['MINUTES'] = pd.to_datetime(df1['TIME'],format='%H:%M:%S').dt.minute

df1.info()
df1
df1.isnull()
# extracting columns for a particular source key
gg = df1.loc[df1['SOURCE_KEY'] == 'Et9kgGMDl729KT4']
gg
gg.info()
gg['DATE'].value_counts()
# checking the daily yield for that particular inverter on a particular date 
gg1 = gg[gg['DATE']=='2020-05-20']
gg1
plt.figure(figsize=(12,8))
plt.plot(gg1['HOUR'],gg1['DAILY_YIELD'] , label = 'yield per hour')
plt.xlabel('time')
plt.ylabel('daily yield')
plt.legend()
plt.show()
# for detailed analysis at 3pm
gg2 = gg1[gg1['HOUR']== 15 ]
gg2
plt.figure(figsize=(12,8))
plt.plot(gg2['MINUTES'],gg2['DAILY_YIELD'] , label = 'yield per minute at 3pm')
plt.xlabel('time')
plt.ylabel('daily yield')
plt.legend()
plt.show()
#for total yield
plt.figure(figsize=(12,8))
plt.plot(gg1['HOUR'],gg1['TOTAL_YIELD'] , label = ' total yield per hour')
plt.xlabel('time')
plt.ylabel('total yield')
plt.legend()
plt.show()
plt.figure(figsize=(12,8))
plt.plot(gg2['MINUTES'],gg2['TOTAL_YIELD'] , label = ' total yield per minute at 3pm')
plt.xlabel('time')
plt.ylabel('total yield')
plt.legend()
plt.show()
#linear regression for Et9kgGMDl729KT4 on 2020-05-20
gg1.info()
X = gg1.iloc[:,9:10]
y = gg1.iloc[:,5]
X.ndim
y.ndim
X.shape
y.shape
plt.scatter(X,y)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
X_train.shape
X_test.shape
y_train.shape
y_test.shape
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
y_pred = lin_reg.predict(X_test)
y_pred
y_test
plt.figure(figsize=(12,8))
plt.plot(X_test,y_test,color ='k',label='actual')
plt.plot(X_test,y_pred,color ='orange',label='predicted')
plt.xlabel('daily yield')
plt.ylabel('time')
plt.legend()
plt.show()
lin_reg.coef_
lin_reg.intercept_
lin_reg.predict([[16.1]])
148.67996174*16.1+1427.386187298328
!pip install flask-ngrok
#creating a website for predicting value of y 
from flask_ngrok import run_with_ngrok
from flask import Flask

app = Flask(__name__)
run_with_ngrok(app)

@app.route('/')
def home():
    return "welcome to ml world"

@app.route('/<float:x>')
def ml(x):
    return(str(lin_reg.predict([[x]])))
app.run()


