import pandas as pd

import warnings 

import numpy as np

from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error

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

df
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



# Categorising holiday as 1 (for True) and 0 (for false)

df['is_holiday'].replace('None',0,inplace=True)

df.is_holiday=df.is_holiday.apply(lambda x: 1 if x!=0 else 0)





df['time']=df['time'].apply(lambda x: int(str(x).split(':')[0]))

df['day']=pd.to_datetime(df['date']).dt.day_name()

train_uniquelist=np.array(df.day.unique())

df['day']=apply_encoding(train_uniquelist,df['day'])



df.head()
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

X=data

y=target



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
xgb = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.07, gamma=0, subsample=0.75,

                           colsample_bytree=1,max_depth=10) 

#increasing n_estimator to 1000 decreases accuracy

#increasing max_depth beyond 7 decreases accuracy

xgb.fit(X_train,y_train)

y_pred=xgb.predict(X_test)

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score

from math import sqrt



rms = sqrt(mean_squared_error(y_pred, y_test))

print("MAE = ",mean_absolute_error(y_pred,y_test))

print('RMS = ',rms)

print(r2_score(y_pred,y_test))

features = pd.DataFrame()

features['feature'] = X_train.columns

features['importance'] = xgb.feature_importances_

features.sort_values(by=['importance'], ascending=True, inplace=True)

features.set_index('feature', inplace=True)

features.plot(kind='barh')

plt.show()
import matplotlib.pyplot as plt

import seaborn as sns

corrmat = df.corr() 

  

f, ax = plt.subplots() 

sns.heatmap(corrmat, ax = ax, cmap ="YlGnBu", linewidths = 0.1) 
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor

clf = RandomForestRegressor(n_estimators=20, max_features='sqrt')

clf = clf.fit(X_train, y_train)

features = pd.DataFrame()

features['feature'] = X_train.columns

features['importance'] = clf.feature_importances_

features.sort_values(by=['importance'], ascending=True, inplace=True)

features.set_index('feature', inplace=True)

features.plot(kind='barh', figsize=(25,25))

plt.show()