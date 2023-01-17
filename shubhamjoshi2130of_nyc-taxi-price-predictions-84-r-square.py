# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


# Any results you write to the current directory are saved as output.

import numpy as np
import pandas as pd
import datetime as datetime
import matplotlib as mp
import matplotlib.pyplot  as plt
import seaborn as sns

taxi_train_df =  pd.read_csv('../input/train.csv', nrows = 1_000_000, parse_dates=["pickup_datetime"])
taxi_train_df.head()
taxi_train_df.isnull().sum()
taxi_train_df=taxi_train_df.loc[taxi_train_df.dropoff_longitude.notnull() & taxi_train_df.dropoff_latitude.notnull(),:]
taxi_train_df.isnull().sum()
import calendar
from math import radians,sin, cos, sqrt, atan2,acos


def get_distance(lat1,lon1,lat2,lon2):
    dist=0
    if((lat1!=lat2) | (lon1!=lon2)):
        slat = radians(lat1)
        slon = radians(lon1)
        elat = radians(lat2)
        elon = radians(lon2)
        dist = 6371.01 * acos(sin(slat)*sin(elat) + cos(slat)*cos(elat)*cos(slon - elon))
    #distance = 6371.01 * acos(sin(lat1)*sin(lat2) + cos(lat1)*cos(lat2)*cos(lon1 - lon2))    
    return dist


# Lets find out the outliers to find these lets first calculate the  euladian distance between laattitude and logitude
# divide total  fare by the distance

taxi_train_df["euc_dis"]=taxi_train_df.apply(lambda df:get_distance(df['pickup_latitude'],df['pickup_longitude'],df['dropoff_latitude'],df['dropoff_longitude']), axis=1)

taxi_train_df["week_day"]=taxi_train_df["pickup_datetime"].dt.day_name()
taxi_train_df["hour"]=taxi_train_df["pickup_datetime"].dt.hour

taxi_train_df.describe()
taxi_train_df[taxi_train_df.euc_dis==0].count()
taxi_train_df[(taxi_train_df.pickup_longitude==taxi_train_df.dropoff_longitude) & (taxi_train_df.pickup_latitude==taxi_train_df.dropoff_latitude)].count()
taxi_train_df[(taxi_train_df.pickup_longitude==taxi_train_df.dropoff_longitude) & (taxi_train_df.pickup_latitude==taxi_train_df.dropoff_latitude)].head()['fare_amount']
taxi_train_df.loc[((taxi_train_df.pickup_longitude==taxi_train_df.dropoff_longitude) & (taxi_train_df.pickup_latitude==taxi_train_df.dropoff_latitude)) | (taxi_train_df.euc_dis==0),'is_completed']=0
taxi_train_df.loc[np.invert((taxi_train_df.pickup_longitude==taxi_train_df.dropoff_longitude) & (taxi_train_df.pickup_latitude==taxi_train_df.dropoff_latitude)) & (taxi_train_df.euc_dis!=0),'is_completed']=1
taxi_train_df.describe()
taxi_train_df.loc[taxi_train_df.is_completed==1 & taxi_train_df.euc_dis.notnull(),'price_per_km']=taxi_train_df['fare_amount']/taxi_train_df['euc_dis']
taxi_train_df.loc[taxi_train_df.is_completed==0 & taxi_train_df.euc_dis.isnull(),'price_per_km']=taxi_train_df['fare_amount']
taxi_train_df.describe()
taxi_train_df.isnull().sum()
taxi_train_df['price_per_km'].quantile(np.arange(0,1.1,.1))
taxi_train_df.loc[taxi_train_df.price_per_km<0,:]
taxi_train_df.loc[taxi_train_df.price_per_km>1.246708e+06,:]
# Lets remove this record as this is a outlier
taxi_train_df=taxi_train_df.loc[taxi_train_df.price_per_km<1.246708e+06,:]
# Lets remove the records with fare_amount as negetive
taxi_train_df=taxi_train_df[taxi_train_df.fare_amount>0]
# Lets remove the records where euc_dis is 0 and fare is greter than 0
taxi_train_df=taxi_train_df.loc[np.invert((taxi_train_df.euc_dis==0) & (taxi_train_df.fare_amount>0)),:]
taxi_train_df['price_per_km'].describe()
taxi_train_df[taxi_train_df.price_per_km>895345]
# We observe that lots of distance are less verry close to 0 which has lead to increase in fare amount, lets remove these records
taxi_train_df=taxi_train_df[(taxi_train_df.euc_dis>.1)]
taxi_train_df['price_per_km'].quantile(np.arange(0,1.1,.1))
# Lets remove all records with price_per_km greater than 2 digits
from math import log10
taxi_train_df=taxi_train_df[(np.log10(taxi_train_df.price_per_km) + 1).astype(int)<=2]

taxi_train_df[np.log10(taxi_train_df.price_per_km).astype(int)>2]
taxi_train_df['price_per_km'].quantile(np.arange(0,1.1,.1))

#visualise_numeric(taxi_train_df,'price_per_km','fare_amount')
sns.boxplot(y='price_per_km',data=taxi_train_df)
sns.distplot(taxi_train_df['price_per_km'],bins=50)
# Lets remove outliers using Inter quartile ranges
Q1 = taxi_train_df['price_per_km'].quantile(0.25)
Q3 = taxi_train_df['price_per_km'].quantile(0.75)
IQR = Q3 - Q1
print(Q1)
print(Q3)
print(IQR)
print(Q1 - 1.5 * IQR)
print((Q3 + 1.5 * IQR))
taxi_train_df_out_rem=taxi_train_df[((taxi_train_df.price_per_km >= (Q1 - 1.5 * IQR)) & (taxi_train_df.price_per_km <= (Q3 + 1.5 * IQR)))]
taxi_train_df_out_rem['price_per_km'].quantile(np.arange(0,1.1,.1))

sns.distplot(taxi_train_df_out_rem['price_per_km'],bins=50)
#visualise_numeric(taxi_train_df,'price_per_km','fare_amount')
sns.boxplot(y='price_per_km',data=taxi_train_df_out_rem)
taxi_train_df_out_rem['price_per_km'].quantile(np.arange(0,1.1,.1))

((1000000-914098)/1000000)*100
sns.distplot(taxi_train_df_out_rem['passenger_count'],bins=50)
# Lets plot the points on xy axis and check if we can apply simple liear regression on it
sns.scatterplot(x="pickup_longitude", y="fare_amount", data=taxi_train_df_out_rem)
# Lets plot the points on xy axis and check if we can apply simple liear regression on it
sns.scatterplot(x="pickup_latitude", y="fare_amount", data=taxi_train_df_out_rem)
# Lets plot the points on xy axis and check if we can apply simple liear regression on it
sns.scatterplot(x="dropoff_longitude", y="fare_amount", data=taxi_train_df_out_rem)
sns.scatterplot(x="dropoff_latitude", y="fare_amount", data=taxi_train_df_out_rem)
sns.scatterplot(x="week_day", y="fare_amount", data=taxi_train_df_out_rem)
sns.barplot(x="hour", y="fare_amount", data=taxi_train_df_out_rem, estimator=sum)
sns.barplot(x="hour", y="fare_amount", data=taxi_train_df_out_rem, estimator=np.mean)
def get_time_labels(x):
    if x>=3 and x<=7:
        return 'EARLY_MORNING'
    elif x>=8 and x<=11:
        return 'MORNING'
    elif x>=12 and x<=17:
        return 'AFTERNOON'
    elif x>=18 and x<=22:
        return 'EVENING'
    elif x==23 or x==0 or (x>=1 and x<=2):
        return 'LATE_EVENING'
    else:
        raise ValueError('Please input values betweenb 0 to 23')
    

    
taxi_train_df_out_rem.loc[:,'DAY_TIME']=taxi_train_df_out_rem.apply(lambda x:get_time_labels(x.hour),axis=1)
taxi_train_df_out_rem.head()
taxi_train_df_noenc=taxi_train_df_out_rem.loc[:,['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','passenger_count','week_day','DAY_TIME','fare_amount']]

# lets do one hot encoding
taxi_train_df_enc=pd.get_dummies(taxi_train_df_noenc, drop_first=True)
taxi_train_df_enc.head()
from sklearn.model_selection import train_test_split

X=taxi_train_df_enc.drop('fare_amount',axis=1)

y=taxi_train_df_enc[['fare_amount']]

X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=.7,random_state=100)

# Lets check if means are comparable
print(y_train.mean(),y_test.mean())

import xgboost as xgb

from xgboost import XGBRegressor

model=XGBRegressor()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)

# Lets evaluate the results

from sklearn.metrics import mean_squared_error,r2_score

mse=mean_squared_error(y_test,y_pred)
r_square=r2_score(y_test,y_pred)
print("mse:",mse)
print("r_square:",r_square)
# parameter grid
parameter_grid={"learning_rate":[0.2,0.6,0.9],"subsample":[0.3,0.6,0.9]}

# specify a model
xgb_model=XGBRegressor(max_depth=2,n_estimators=200)

# setup a GridSearch
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

model_cv=GridSearchCV(estimator=xgb_model,param_grid=parameter_grid,cv=3,scoring="r2")

model_cv.fit(X_train,y_train)

cv_results=pd.DataFrame(model_cv.cv_results_)
cv_results
xgb_model_final=XGBRegressor(max_depth=2,n_estimators=200,learning_rate=0.6,subsample=0.9)
xgb_model_final.fit(X_train,y_train)

y_pred=xgb_model_final.predict(X_test)
mse=mean_squared_error(y_test,y_pred)
r_square=r2_score(y_test,y_pred)
print("mse:",mse)
print("r_square:",r_square)
taxi_test_df =  pd.read_csv('../input/test.csv', parse_dates=["pickup_datetime"])
taxi_test_df=taxi_test_df.loc[taxi_test_df.dropoff_longitude.notnull() & taxi_test_df.dropoff_latitude.notnull(),:]
taxi_test_df["week_day"]=taxi_test_df["pickup_datetime"].dt.day_name()
taxi_test_df["hour"]=taxi_test_df["pickup_datetime"].dt.hour
taxi_test_df.loc[:,'DAY_TIME']=taxi_test_df.apply(lambda x:get_time_labels(x.hour),axis=1)
taxi_test_df_noenc=taxi_test_df.loc[:,['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','passenger_count','week_day','DAY_TIME']]
taxi_test_df_key=taxi_test_df['key']
taxi_test_df_enc=pd.get_dummies(taxi_test_df_noenc, drop_first=True)
y_pred=xgb_model_final.predict(taxi_test_df_enc)

result_df=pd.DataFrame({"key":taxi_test_df_key,"fare_amount":y_pred})
result_df.head()