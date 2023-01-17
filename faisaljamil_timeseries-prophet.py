# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_pjm=pd.read_csv('/kaggle/input/hourly-energy-consumption/PJME_hourly.csv',index_col=[0],parse_dates=[0])
df_pjm.head()
df_pjm.info()
print(df_pjm.isnull().sum())
data=df_pjm
plt.figure(figsize=(16,8))
sns.lineplot(data.index,'PJME_MW',data=data)
def create_feature(df,label=None):
    df=df.copy()
    df['date']=df.index
    df['hour']=df['date'].dt.hour
    df['dayofweek']=df['date'].dt.dayofweek
    df['quarter']=df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.weekofyear
    
    
    X = df[['hour','dayofweek','quarter','month','year',
           'dayofyear','dayofmonth','weekofyear']]
        
    if label:
        y=df[label]
        return X,y
    return X

X,y=create_feature(data,label='PJME_MW')

X.tail()
trainData=pd.concat([X,y],axis=1)
trainData.head()
fig=sns.pairplot(trainData,hue='hour',kind="line",x_vars=['hour','dayofweek','year','month'],y_vars='PJME_MW')

sns.set(font_scale=2)
fig=sns.pairplot(trainData,x_vars=['weekofyear'],hue="hour", palette="ch:r=-.5,l=.75",y_vars='PJME_MW', height=12.5, markers="o")
plt.xlabel('Time Step (Week)') 
plt.ylabel('Energy Consumption (MW)')
fig.savefig('week.png')
sns.set(font_scale=2)
fig=sns.pairplot(trainData,x_vars=['hour'],hue="hour", palette="ch:r=-.5,l=.75",y_vars='PJME_MW', height=12.5, markers="o")
plt.xlabel('Time Step (Hour)') 
plt.ylabel('Energy Consumption (MW)')
fig.savefig('hour.png')

sns.pairplot(trainData,x_vars=['dayofweek'],hue="hour", palette="ch:r=-.5,l=.75",y_vars='PJME_MW',height=12.5)
plt.xlabel('Time Step (Day)') 
plt.ylabel('Energy Consumption (MW)')
fig.savefig('Day.png')
sns.pairplot(trainData, palette="ch:r=-.5,l=.75",hue="hour",x_vars=['year'],y_vars='PJME_MW',height=12.5)
plt.xlabel('Time Step (Year)') 
plt.ylabel('Energy Consumption (MW)')
fig.savefig('year.png')
sns.pairplot(trainData, palette="ch:r=-.5,l=.75",hue="hour",x_vars=['month'],y_vars='PJME_MW',height=12.5)
plt.xlabel('Time Step (Month)') 
plt.ylabel('Energy Consumption (MW)')
fig.savefig('month.png')
sns.pairplot(trainData, palette="ch:r=-.5,l=.75",hue="hour",x_vars=['quarter'],y_vars='PJME_MW',height=12.5)
plt.xlabel('Time Step (Quarter)') 
plt.ylabel('Energy Consumption (MW)')
fig.savefig('quarter.png')
split_date = '01-Jan-2015'
pjme_train = data.loc[data.index <= split_date].copy()
pjme_test=data.loc[data.index>split_date].copy()

pjme_train.reset_index().rename(columns={'Datetime':'ds','PJME_MW':'y'}).head()
#df['ds']=pjme_train['Datetime'].copy()
from fbprophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error

model=Prophet()
model.fit(pjme_train.reset_index().rename(columns={'Datetime':'ds','PJME_MW':'y'}))
Prediction=model.predict(pjme_test.reset_index().rename(columns={'Datetime':'ds','PJME_MW':'y'}))
model.plot(Prediction)
mean_absolute_error(pjme_test['PJME_MW'],Prediction['yhat'])