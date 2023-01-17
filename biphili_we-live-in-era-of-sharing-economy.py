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
import pandas as pd

import matplotlib.pyplot as plt

import numpy as np 

import math 
pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)
df1=pd.read_csv('../input/bike-sharing-demand/train.csv',parse_dates=['datetime'],index_col=0)

df_test=pd.read_csv('../input/bike-sharing-demand/test.csv',parse_dates=['datetime'],index_col=0)

df1.head()
#df1.info()

def add_feature(df1):

    df1['year']=df1.index.year

    df1['month']=df1.index.month

    df1['dayofmonth']=df1.index.day

    df1['dayofweek']=df1.index.dayofweek

    df1['hour']=df1.index.hour
add_feature(df1)

add_feature(df_test)

df1.head()
df1.shape
df =df1.copy()
df = df.rename(columns={'count':'demand'})
df.reset_index(drop=True, inplace=True)

df = df.drop(['casual','registered'],axis=1)

df.head()
df.isnull().sum()
df.hist(figsize = (15,10))

plt.tight_layout()

plt.show()
plt.figure(figsize=(15,10))

plt.subplot(2,2,1,)

plt.title('Temperature Vs Demand')

plt.xlabel('Temperature')

plt.ylabel('Demand')

plt.scatter(df['temp'],df['demand'],s=2,c='g')



plt.subplot(2,2,2)

plt.title('Abs Temperature Vs Demand')

plt.xlabel('Abs Temperature')

plt.ylabel('Demand')

plt.scatter(df['atemp'],df['demand'],s=2,c='b')



plt.subplot(2,2,3)

plt.title('Humidity Vs Demand')

plt.xlabel('Humidity')

plt.ylabel('Demand')

plt.scatter(df['humidity'],df['demand'],c='r')



plt.subplot(2,2,4)

plt.title('Windspeed Vs Demand')

plt.xlabel('Windspeed')

plt.ylabel('Demand')

plt.scatter(df['windspeed'],df['demand'],c='c')



plt.tight_layout()



pass
colors = ['g','r','m','b']



plt.figure(figsize=(15,10))

plt.subplot(3,3,1,)

plt.title('Seasons Vs Demand')

plt.xlabel('Seasons')

plt.ylabel('Demand')

cat_list=df['season'].unique()

cat_average=df.groupby('season').mean()['demand']

plt.bar(cat_list,cat_average,color=colors)



plt.subplot(3,3,2,)

plt.title('Holiday Vs Demand')

plt.xlabel('Holiday')

plt.ylabel('Demand')

cat_list=df['holiday'].unique()

cat_average=df.groupby('holiday').mean()['demand']

plt.bar(cat_list,cat_average,color=colors)



plt.subplot(3,3,3,)

plt.title('Working Day Vs Demand')

plt.xlabel('Working Day')

plt.ylabel('Demand')

cat_list=df['workingday'].unique()

cat_average=df.groupby('workingday').mean()['demand']

plt.bar(cat_list,cat_average,color=colors)



plt.subplot(3,3,4,)

plt.title('Weather Vs Demand')

plt.xlabel('Weather')

plt.ylabel('Demand')

cat_list=df['weather'].unique()

cat_average=df.groupby('weather').mean()['demand']

plt.bar(cat_list,cat_average,color=colors)



plt.subplot(3,3,5,)

plt.title('Year Vs Demand')

plt.xlabel('Year')

plt.ylabel('Demand')

cat_list=df['year'].unique()

cat_average=df.groupby('year').mean()['demand']

plt.bar(cat_list,cat_average,color=colors)



plt.subplot(3,3,6,)

plt.title('Month Vs Demand')

plt.xlabel('Month')

plt.ylabel('Demand')

cat_list=df['month'].unique()

cat_average=df.groupby('month').mean()['demand']

plt.bar(cat_list,cat_average,color=colors)



plt.subplot(3,3,7,)

plt.title('Day Of Month Vs Demand')

plt.xlabel('Day Of Month')

plt.ylabel('Demand')

cat_list=df['dayofmonth'].unique()

cat_average=df.groupby('dayofmonth').mean()['demand']

plt.bar(cat_list,cat_average,color=colors)



plt.subplot(3,3,8,)

plt.title('Day Of Week Vs Demand')

plt.xlabel('Day Of Week')

plt.ylabel('Demand')

cat_list=df['dayofweek'].unique()

cat_average=df.groupby('dayofweek').mean()['demand']

plt.bar(cat_list,cat_average,color=colors)



plt.subplot(3,3,9,)

plt.title('Hour Vs Demand')

plt.xlabel('Hour')

plt.ylabel('Demand')

cat_list=df['hour'].unique()

cat_average=df.groupby('hour').mean()['demand']

plt.bar(cat_list,cat_average,color=colors)





plt.tight_layout()

pass
df_prep = df.copy()

#df_prep
df_prep['demand'].describe()
df_prep['demand'].quantile([0.05,0.1,0.15,0.9,0.99])
correlation = df_prep[['temp','atemp','humidity','demand']].corr()

correlation
#df.head()
df_prep = df_prep.drop(['atemp','holiday','workingday','year','dayofmonth','dayofweek'],axis=1)

df_prep.head()

df2 = pd.to_numeric(df_prep['demand'],downcast='float')

plt.acorr(df2,maxlags=12)

pass
d1 = df_prep['demand']

d2 = np.log(d1)



plt.figure(figsize=(15,10))

plt.subplot(1,2,1,)

plt.title('Actual Demand Plot')

plt.xlabel('Frequncy')

plt.ylabel('Demand')

d1.hist(rwidth=0.9)



plt.subplot(1,2,2,)

plt.title('Log Transform Demand Plot')

plt.xlabel('Frequncy')

plt.ylabel('Demand')

d2.hist(rwidth=0.9)



pass

df_prep['demand'] = np.log(df_prep['demand'])

#df_prep
t_1 = df_prep['demand'].shift(+1).to_frame()

t_1.columns = ['t-1']



t_2 = df_prep['demand'].shift(+2).to_frame()

t_2.columns = ['t-2']



t_3 = df_prep['demand'].shift(+3).to_frame()

t_3.columns = ['t-3']
df_prep_lag = pd.concat([df_prep,t_1,t_2,t_3],axis=1)

df_prep_lag = df_prep_lag.dropna()

df_prep_lag.head()
df_prep_lag.dtypes
df_prep_lag['season'] = df_prep_lag['season'].astype('category')

df_prep_lag['weather'] = df_prep_lag['weather'].astype('category')

df_prep_lag['month'] = df_prep_lag['month'].astype('category')

df_prep_lag['hour'] = df_prep_lag['hour'].astype('category')

dummy_df = pd.get_dummies(df_prep_lag,drop_first=True)

dummy_df.head()

#dummy_df.shape
y = df_prep_lag[['demand']]

X = df_prep_lag.drop(['demand'],axis=1)
# Creating the training set at 70%

tr_size = 0.7*len(X)

tr_size = int(tr_size)



X_train = X.values[0:tr_size]

X_test = X.values[tr_size:len(X)]



y_train = y.values[0:tr_size]

y_test = y.values[tr_size:len(y)]
from sklearn.linear_model import LinearRegression 



std_reg = LinearRegression()

std_reg.fit(X_train,y_train)
r2_train = std_reg.score(X_train,y_train)

r2_test = std_reg.score(X_test,y_test)

print('R Suared Error for Train set:',r2_train)

print('R Suared Error for Test set:',r2_test)
# Create Predictions 

y_predict = std_reg.predict(X_test)

from sklearn.metrics import mean_squared_error

rmse = math.sqrt(mean_squared_error(y_test,y_predict))

print('RMSE of the model:',rmse)
y_test_e = []

y_predict_e = []





for i in range(0,len(y_test)):

    y_test_e.append(math.exp(y_test[i]))

    y_predict_e.append(math.exp(y_predict[i]))

    

# Calculate the sum

log_sq_sum = 0.0



for i in range(0,len(y_test_e)):

    log_a = math.log(y_test_e[i] +1)

    log_p = math.log(y_predict_e[i] +1)

    log_diff = (log_p - log_a)**2

    log_sq_sum = log_sq_sum + log_diff

    

rmsle = math.sqrt(log_sq_sum/len(y_test))

print(rmsle)
