# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from math import sqrt

from sklearn.metrics import r2_score

import xgboost as xgb

import featuretools as ft

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# PANDA LIBRARY'SI  UZERINDEN   DATAFRAME OBJESI YARATARAK ILK 1M ROWU OKUYORUM

df_train=pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/train.csv", sep=',', lineterminator='\n',nrows=1000000)

df_train.rename(columns={'passenger_count\r':'passenger_count'}, inplace=True)

print("df_train size ",df_train.info()) 

df_train.info()



print( df_train.head() )
# IS NULL CHECK

print("NULL CHECK")

print( df_train.columns[df_train.isnull().any()].tolist() )



print("UNIQUE CHECK")

# UNIQUE CHECK

for  d in df_train.columns:

    print(d+"  "+str( df_train[d].is_unique ) )

    

    

print("LIST OF UNIQUE VALUES")

print( "passenger_count "+str( df_train.passenger_count.unique() ) )

print( "pickup_longitude  "+str( df_train.pickup_longitude.unique() ) )





print("GET INDEX")

print(df_train.index)



# general info



df_train[['fare_amount','passenger_count']].describe()
# CHECK INVALID  DATA

sns.relplot(x="passenger_count", y="fare_amount", data=df_train);
#  FILTER AND RECHECK

sns.relplot(x="passenger_count", y="fare_amount", data=df_train[(df_train.passenger_count <20) & (df_train.passenger_count>0)]);
#FARE AMOUNT KONTOLLERI

sns.boxplot(x=df_train['fare_amount'])

#FARE AMOUNT KONTOLLERI

df_train_2=df_train[(df_train.fare_amount<15) & (df_train.fare_amount>0)]

sns.boxplot(x=df_train_2['fare_amount'])
# HEAT MAP CORRELATION ILE ILGILI

plt.figure(figsize=(15,7))

sns.heatmap(df_train.corr(),annot=True)

# feauture engineering

# ENLEM VE BOYLAM KONTROLLERI

def add_travel_vector_features(df):

    df['abs_diff_longitude'] = (df.dropoff_longitude - df.pickup_longitude).abs()

    df['abs_diff_latitude'] = (df.dropoff_latitude - df.pickup_latitude).abs()



add_travel_vector_features(df_train)



sns.relplot(x="abs_diff_latitude", y="abs_diff_longitude", data=df_train);
#CLEAN DATA  "abs_diff_latitude", y="abs_diff_longitude",



def clean_df(df):

    return df[

              (df.fare_amount<15) & (df.fare_amount>0) & 

              (df.abs_diff_latitude<5) & (df.abs_diff_longitude<5) &

              (df.passenger_count > 0) & (df.passenger_count < 10)

             ]





df_train_2 = clean_df(df_train).dropna()

print("Clean data completed")

df_train_2.info()

# HEAT MAP CORRELATION ILE ILGILI

plt.figure(figsize=(15,7))

sns.heatmap(df_train_2.corr(),annot=True)
# feauture engineering

# calculate distance

def  getDistance(lat1,lon1,lat2,lon2):

    R = 6373.0

    

    dlon = lon2 - lon1

    dlat = lat2 - lat1

    a = (np.sin(dlat/2))**2 + np.cos(lat1) * np.cos(lat2) * (np.sin(dlon/2))**2

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c

    return distance



df_train_2["distance"]=getDistance(df_train_2.pickup_latitude,df_train_2.pickup_longitude,df_train_2.dropoff_latitude,df_train_2.dropoff_longitude)





# HEAT MAP CORRELATION ILE ILGILI

plt.figure(figsize=(15,7))

sns.heatmap(df_train_2.corr(),annot=True)





#create pickupdate workday or weekend

df_train_2['date_pickup']=pd.to_datetime(df_train_2['pickup_datetime'])

WeekValue_dict = {0:0,1: 0, 2: 0, 3: 0, 4: 0,5: 1, 6: 1}

df_train_2['WeekValue'] = df_train_2['date_pickup'].dt.dayofweek.map(WeekValue_dict)

 

# seperate month - day - time values

df_train_2['year'] = df_train_2['date_pickup'].dt.year

df_train_2['month'] = df_train_2['date_pickup'].dt.month

df_train_2['day'] = df_train_2['date_pickup'].dt.day

df_train_2['hour'] = df_train_2['date_pickup'].dt.hour





day_int_dict = {6:0,7:0,8:0,9:0,10:0,11:0,12:1,13:1,14:2,15:2,16:2,17:2,18:2,0:4,1:4,2:4,3:4,4:4,5:4,19:3,

20:3,21:3,22:3,23:3}

df_train_2['day_int'] = df_train_2['hour'].map(day_int_dict)

df_train_2['day_name'] = df_train_2['date_pickup'].dt.day_name()   





season_dict = {12:0,1: 0, 2: 0, 3: 1, 4: 1,5: 1, 6: 2, 7: 2, 8: 2,9: 3, 10: 3, 11: 3}

df_train_2['season'] = df_train_2['month'].map(season_dict)





# encoding day_name char --> int



lb_make = LabelEncoder()

df_train_2["day_name_num"] = lb_make.fit_transform(df_train_2["day_name"])



df_train_2.head()
# create  column to define day interval ( morning:0, noon:1 , afternoon:2 , evening:3 ,midnight:4)

# create  column to define week value 0: workday 1:weekend

g=sns.lineplot(x="day_int", y="fare_amount",hue='WeekValue',ci=None, data=df_train_2, estimator=np.sum)

#put legends outside of graph

g.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)
# create  column to define week value 0: workday 1:weekend

sns.barplot(x="WeekValue", y="fare_amount", data=df_train_2, estimator=sum)
#day_int ( morning:0, noon:1 , afternoon:2 , evening:3 ,midnight:4)

g=sns.barplot(x="day_int", y="fare_amount",hue='day_name',hue_order=['Monday', 'Tuesday', 'Wednesday','Thursday','Friday','Saturday','Sunday'],ci=None,data=df_train_2,estimator=np.sum)

#put legends outside of graph

g.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)
#0 winter ,1 april , 2 summer,3 Autumn

g=sns.barplot(x="season", y="fare_amount",hue='day_name',hue_order=['Monday', 'Tuesday', 'Wednesday','Thursday','Friday','Saturday','Sunday'],ci=None,data=df_train_2,estimator=np.sum)

#put legends outside of graph

g.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)
# PREDICTION ALGORITHM





chunk_list = [] 

df_train_4 = pd.DataFrame(columns=['passenger_count', 'distance', 'WeekValue', 'year', 'day_int', 'season', 'day_name_num',

        'abs_diff_longitude','abs_diff_latitude','fare_amount'])

#Cross-validation 

params ={

    # Parameters that we are going to tune.

    'n_estimators':4,

    'max_depth':6, #Result of tuning with CV

    'eta':0.05, #Result of tuning with CV

    #'subsample': 1, #Result of tuning with CV

    #'colsample_bytree': 0.8, #Result of tuning with CV

    # Other parameters

    #'objective':'reg:linear',

    #'eval_metric':'rmse',

    #'silent': 1

}


def index_marks(nrows, chunk_size):

    return range(1 * chunk_size, (nrows // chunk_size + 1) * chunk_size, chunk_size)





def split(dfm, chunk_size):

    indices = index_marks(dfm.shape[0], chunk_size)

    return np.split(dfm, indices)



model = xgb.XGBRegressor(params=params)

chunk_idx = 0

df_train_3 = pd.DataFrame()

chunks = split(df_train_2, 50000)

for c in chunks:

    

    if chunk_idx > 0:

        df_train_3 =  df_train_3.append(c)

    else:

        df_train_3=c

        

    print (len(df_train_3))

    lister_4=['passenger_count', 'distance', 'WeekValue', 'year', 'day_int', 'season', 'day_name_num','abs_diff_longitude',

       'abs_diff_latitude']

    lister_5=['fare_amount']    

    

    X_train, X_test, y_train, y_test= train_test_split(df_train_3[lister_4],df_train_3[lister_5], test_size=0.1,

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
#AUTO FEATURE ENG.

df_train_2_1=df_train_2[:1000]

# creating and entity set 'es'

es = ft.EntitySet(id = 'All_taxi_fare')



# adding a dataframe 

es.entity_from_dataframe(entity_id = 'taxi_all', dataframe = df_train_2_1, index = 'key')





es.normalize_entity(base_entity_id='taxi_all', new_entity_id='tax_dates', index = 'pickup_datetime', 

additional_variables = ['pickup_longitude', 'pickup_latitude','dropoff_longitude', 'dropoff_latitude', 'passenger_count'])



print(es)





feature_matrix, feature_names = ft.dfs(entityset=es, 

target_entity = 'taxi_all', 

max_depth = 2, 

verbose = 1, 

n_jobs = 3)





print(feature_matrix.columns)
feature_matrix.head()
# HEAT MAP CORRELATION ILE ILGILI

plt.figure(figsize=(300,5))

sns.heatmap(feature_matrix.corr().loc[['fare_amount'],:],annot=True)







c_matrix=feature_matrix.corr()["fare_amount"]

for  indexer,f in enumerate(c_matrix):

    if f!=1 and f>=0.3:

        print(c_matrix.index[indexer])

        print(f)

                
# NORMAL DISTRIBUTION



plt.hist(df_train.fare_amount, bins=10)

plt.ylabel('frequency')

plt.show()





plt.hist(df_train_2.fare_amount, bins=10)

plt.ylabel('frequency')

plt.show()