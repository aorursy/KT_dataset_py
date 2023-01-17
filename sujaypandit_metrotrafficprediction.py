import pandas as pd

import warnings 

import numpy as np

from sklearn import linear_model

from sklearn.preprocessing import PolynomialFeatures

import xgboost

warnings.filterwarnings('ignore')

df=pd.read_csv('../input/Train.csv')

dftest=pd.read_csv('../input/Test.csv')

df.head(2)
#Separating date and time columns

df['date_time']=pd.to_datetime(df.date_time)

df['date']=df['date_time'].dt.date

df['time']=df['date_time'].dt.time

dftest['date_time']=pd.to_datetime(dftest.date_time)

dftest['date']=dftest['date_time'].dt.date

dftest['time']=dftest['date_time'].dt.time
dupdf=df.copy()

dupdf.drop(['weather_description','weather_type'],axis=1,inplace=True)

print("Length of duplicated rows= ",len(df[dupdf.duplicated()].index)) # =0 implies no rows are completely repeated even if their time matches

list(df['weather_description'][:20].groupby(df['weather_type']))

#Weather description shouldn't be used in model fit

#Too descriptive adds classes and less value
def apply_encoding(train_uniquelist,col):

    col=col.apply(lambda x: np.where(train_uniquelist==x)[0][0]) #Much faster than for loops

    return pd.Series(col)

train_uniquelist=np.array(df.weather_type.unique())

df.weather_type=apply_encoding(train_uniquelist,df.weather_type)

dftest.weather_type=apply_encoding(train_uniquelist,dftest.weather_type)
# Categorising holiday as 1 (for True) and 0 (for false)

df['is_holiday'].replace('None',0,inplace=True)

df.is_holiday=df.is_holiday.apply(lambda x: 1 if x!=0 else 0)

dftest['is_holiday'].replace('None',0,inplace=True)

dftest.is_holiday=dftest.is_holiday.apply(lambda x: 1 if x!=0 else 0)


df['time']=df['time'].apply(lambda x: int(str(x).split(':')[0]))

df['day']=pd.to_datetime(df['date']).dt.day_name()

train_uniquelist=np.array(df.day.unique())

df['day']=apply_encoding(train_uniquelist,df['day'])

dftest['time']=dftest['time'].apply(lambda x: int(str(x).split(':')[0]))

dftest['day']=pd.to_datetime(dftest['date']).dt.day_name()

dftest['day']=apply_encoding(train_uniquelist,dftest['day'])

## Normalizing data

def normalize(da,cols):

    for i in cols:

        for j in range(len(i)):

            

            if(da[i].max()>1):

                da[i]=da[i]/da[i].max()

    return da
data=df[['day','time','is_holiday','air_pollution_index','humidity','wind_speed','wind_direction','visibility_in_miles','dew_point','temperature','rain_p_h','snow_p_h','clouds_all','weather_type']]

data=normalize(data,data.columns[3:13])

data=pd.DataFrame(data[['day','time','is_holiday','air_pollution_index','wind_speed','visibility_in_miles','temperature','rain_p_h','snow_p_h','clouds_all','weather_type']])

target=df[['traffic_volume']]

X=data.values

y=target.values

xgb = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,

                           colsample_bytree=1,max_depth=7) 

data.head()
data=df[['day','time','is_holiday','air_pollution_index','humidity','wind_speed','wind_direction','visibility_in_miles','dew_point','temperature','rain_p_h','snow_p_h','clouds_all','weather_type']]

data=normalize(data,data.columns[3:13])

data=pd.DataFrame(data[['day','time','is_holiday','air_pollution_index','wind_speed','visibility_in_miles','temperature','rain_p_h','snow_p_h','clouds_all','weather_type']])

target=df[['traffic_volume']]

X=data.values

y=target.values

xgb = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,

                           colsample_bytree=1,max_depth=7) 

#increasing n_estimator to 1000 decreases accuracy

#increasing max_depth beyond 7 decreases accuracy

xgb.fit(X,y)

testData=dftest[['day','time','is_holiday','air_pollution_index','humidity','wind_speed','wind_direction','visibility_in_miles','dew_point','temperature','rain_p_h','snow_p_h','clouds_all','weather_type']]

testData=normalize(testData,testData.columns[3:13])

testData=pd.DataFrame(testData[['day','time','is_holiday','air_pollution_index','wind_speed','visibility_in_miles','temperature','rain_p_h','snow_p_h','clouds_all','weather_type']])

x=testData.values



Y = xgb.predict(X) #x testing on training data

for i in range(len(Y)):

    Y[i]=abs(int(Y[i]))
finaldf=df[['date_time']]#dftest[['date_time']]

finaldf['traffic_volume']=Y

finaldf.to_csv("XG_Regression.csv", sep=',',index=False)
