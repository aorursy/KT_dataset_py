

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib

import datetime as dt

import xgboost as xgb

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from sklearn.metrics import mean_squared_error

from math import sqrt

from sklearn.preprocessing import StandardScaler

import os

print(os.listdir("../input"))

pd.options.mode.chained_assignment = None

pd.set_option('display.max_columns', 500)

# Any results you write to the current directory are saved as output.





from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.metrics import r2_score

df_train=pd.read_csv("../input/train.csv", sep=',', lineterminator='\n',infer_datetime_format=True,nrows=50000000,

chunksize=1000000)

#print("df_train size ",df_train.info()) 



df_test=pd.read_csv("../input/test.csv", sep=',', lineterminator='\n')

df_test.rename(columns={'passenger_count\r':'passenger_count'}, inplace=True)

#print("df_test size ",df_test.info()) 
# calculate distance

def  getDistance(lat1,lon1,lat2,lon2):

    R = 6373.0

    

    dlon = lon2 - lon1

    dlat = lat2 - lat1

    a = (np.sin(dlat/2))**2 + np.cos(lat1) * np.cos(lat2) * (np.sin(dlon/2))**2

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c

    return distance





def clean_df(df):

    return df[(df.fare_amount > 0) & 

            (df.pickup_longitude > -80) & (df.pickup_longitude < -70) &

            (df.pickup_latitude > 35) & (df.pickup_latitude < 45) &

            (df.dropoff_longitude > -80) & (df.dropoff_longitude < -70) &

            (df.dropoff_latitude > 35) & (df.dropoff_latitude < 45) &

            (df.passenger_count > 0) & (df.passenger_count < 10)]



df_test=pd.read_csv("../input/test.csv", sep=',', lineterminator='\n')

df_test.rename(columns={'passenger_count\r':'passenger_count'}, inplace=True)

df_test_2=df_test





df_test_2["distance"]=getDistance(df_test_2.pickup_latitude,df_test_2.pickup_longitude,df_test_2.dropoff_latitude,df_test_2.dropoff_longitude)

print("test distance completed")









df_test_2['date_pickup']=pd.to_datetime(df_test_2['pickup_datetime'])

#df_train_2['WeekValue'] = np.where(df_train_2['date_pickup'].dayofweek<5, 0, 1) 

print("test conversion completed")



WeekValue_dict = {0:0,1: 0, 2: 0, 3: 0, 4: 0,5: 1, 6: 1}

df_test_2['WeekValue'] = df_test_2['date_pickup'].dt.dayofweek.map(WeekValue_dict)

print("test WeekValue completed")   





# seperate month - day - time values

df_test_2['year'] = df_test_2['date_pickup'].dt.year

df_test_2['month'] = df_test_2['date_pickup'].dt.month

df_test_2['day'] = df_test_2['date_pickup'].dt.day

#df_test_2['hour'] = df_test_2['pickup_datetime'].dt.hour

df_test_2['hour'] = df_test_2['date_pickup'].dt.hour

print(" test dates completed")



day_int_dict = {6:0,7:0,8:0,9:0,10:0,11:0,12:1,13:1,14:2,15:2,16:2,17:2,18:2,0:4,1:4,2:4,3:4,4:4,5:4,19:3,

20:3,21:3,22:3,23:3}

df_test_2['day_int'] = df_test_2['hour'].map(day_int_dict)

df_test_2['day_name'] = df_test_2['date_pickup'].dt.day_name()



season_dict = {12:0,1: 0, 2: 0, 3: 1, 4: 1,5: 1, 6: 2, 7: 2, 8: 2,9: 3, 10: 3, 11: 3}

df_test_2['season'] = df_test_2['month'].map(season_dict)

print("test 3rd completed")





lb_make = LabelEncoder()

df_test_2["day_name_num"] = lb_make.fit_transform(df_test_2["day_name"])



df_test_3=df_test_2[['passenger_count', 'distance', 'WeekValue', 'year','month','day_int','season','day_name_num',"key",'pickup_longitude',

       'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']]







chunk_list = [] 

df_train_4 = pd.DataFrame(columns=['passenger_count', 'distance', 'WeekValue', 'year', 'day_int', 'season', 'day_name_num','pickup_longitude',

       'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude','fare_amount'])

#Cross-validation 

params ={

    # Parameters that we are going to tune.

    'n_estimators':100,

    'max_depth':8, #Result of tuning with CV

    'eta':0.01, #Result of tuning with CV

    #'subsample': 1, #Result of tuning with CV

    #'colsample_bytree': 0.8, #Result of tuning with CV

    # Other parameters

    #'objective':'reg:linear',

    #'eval_metric':'rmse',

    #'silent': 1

}





model = xgb.XGBRegressor(params=params)

chunk_idx = 0

# Each chunk is in df format

for chunk in df_train:  

    # perform data filtering 

    #chunk_filter = chunk_preprocessing(chunk)

    chunk.rename(columns={'passenger_count\r':'passenger_count'}, inplace=True)

    

    have_pass =  chunk['passenger_count']>0

    have_money=  chunk['fare_amount']>0

    df_train_2 = chunk[have_pass & have_money]

    

    

    df_train_2 = clean_df(df_train_2)

    print("Clean data completed")



    """

    Q1 = df_train_2['fare_amount'].quantile(0.25)

    Q3 = df_train_2['fare_amount'].quantile(0.75)

    IQR = Q3 - Q1

    print(IQR)

    df_train_3 = df_train_2[~((df_train_2['fare_amount'] < (Q1 - 1.5 * IQR)) |(df_train_2['fare_amount'] > (Q3 + 1.5 * IQR)))]

    df_train_3.shape

    df_train_2=df_train_3

    """

    

    df_train_2["distance"]=getDistance(df_train_2.pickup_latitude,df_train_2.pickup_longitude,df_train_2.dropoff_latitude,df_train_2.dropoff_longitude)

    print("train distance completed")

    

    # create column to define day is weekday or weekend    

    df_train_2['date_pickup']=pd.to_datetime(df_train_2['pickup_datetime'],format='%Y-%m-%d %H:%M:%S UTC')

    #df_train_2['WeekValue'] = np.where(df_train_2['date_pickup'].dayofweek<5, 0, 1) 

    print("train conversion completed")



    WeekValue_dict = {0:0,1: 0, 2: 0, 3: 0, 4: 0,5: 1, 6: 1}

    df_train_2['WeekValue'] = df_train_2['date_pickup'].dt.dayofweek.map(WeekValue_dict)

    print("train WeekValue completed")   





    # seperate month - day - time values

    df_train_2['year'] = df_train_2['date_pickup'].dt.year

    df_train_2['month'] = df_train_2['date_pickup'].dt.month

    df_train_2['day'] = df_train_2['date_pickup'].dt.day

    #df_train_2['hour'] = df_train_2['pickup_datetime'].dt.hour

    df_train_2['hour'] = df_train_2['date_pickup'].dt.hour

    print(" train dates completed")

    

    

    # create  column to define day interval ( morning:0, noon:1 , afternoon:2 , evening:3 ,midnight:4)

    day_int_dict = {6:0,7:0,8:0,9:0,10:0,11:0,12:1,13:1,14:2,15:2,16:2,17:2,18:2,0:4,1:4,2:4,3:4,4:4,5:4,19:3,

    20:3,21:3,22:3,23:3}

    df_train_2['day_int'] = df_train_2['hour'].map(day_int_dict)

    df_train_2['day_name'] = df_train_2['date_pickup'].dt.day_name()

    #0 winter ,1 april , 2 summer,3 Autumn

    season_dict = {12:0,1: 0, 2: 0, 3: 1, 4: 1,5: 1, 6: 2, 7: 2, 8: 2,9: 3, 10: 3, 11: 3}

    df_train_2['season'] = df_train_2['month'].map(season_dict)



    print("train 3rd completed")

    

    # encoding day_name char --> int



    lb_make = LabelEncoder()

    df_train_2["day_name_num"] = lb_make.fit_transform(df_train_2["day_name"])

    

    #chunk_list.append(df_train_2[['passenger_count', 'distance', 'WeekValue', 'year','month','day_int','season','day_name_num','fare_amount','pickup_longitude',

    #  'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']])

    if chunk_idx > 0:

        df_train_4.append(df_train_2)

        #df_train_4=df_train_2

    else:

        df_train_4=df_train_2

    lister_4=['passenger_count', 'distance', 'WeekValue', 'year', 'day_int', 'season', 'day_name_num','pickup_longitude',

       'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']

    lister_5=['fare_amount']



    X_train, X_test, y_train, y_test= train_test_split(df_train_4[lister_4],df_train_4[lister_5], test_size=0.1,

    random_state=42)

    

    if chunk_idx > 0: # not load in first run

        model.fit(X_train, y_train, xgb_model='model_1.model')

        model.save_model('model_1.model')

    else:

        model.fit(X_train, y_train)

        model.save_model('model_1.model')

    chunk_idx = chunk_idx + 1

    rmse = sqrt(mean_squared_error(y_test, model.predict(X_test)))

    print("RMSE RESULT ",rmse)

    # evaluate predictions

    accuracy = r2_score(y_test, model.predict(X_test))

    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    

    #kfold = KFold(n_splits=2, random_state=7)

    #results = cross_val_score(model, X_train, y_train, cv=kfold)

    #print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

    print(1)

    predictions = model.predict(df_test_3[lister_4])

    my_submission = pd.DataFrame( { 'key': df_test_3.key,'fare_amount': predictions.round(2) } )

# concat the list into dataframe 

print("here")

#df_concat = pd.concat(chunk_list)

#print(df_concat.info())
#print(df_test_3.head())

#print(df_test_3.info())

"""

Index(['key', 'fare_amount', 'pickup_datetime', 'pickup_longitude',

       'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',

       'passenger_count', 'distance', 'date_pickup', 'WeekValue', 'year',

       'month', 'day', 'hour', 'day_int', 'day_name', 'season',

       'day_name_num'],

      dtype='object')

"""
# you could use any filename. We choose submission here

my_submission.to_csv('sample_submission.csv', index=False)

print("Writing complete")


