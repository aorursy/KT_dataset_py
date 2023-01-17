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
import featuretools as ft
data = ft.demo.load_mock_customer()
customers_df = data["customers"]
customers_df.head()
sessions_df = data['sessions']
sessions_df.head(5)
transactions_df = data["transactions"]
transactions_df.head(5)
# Create new entityset

es = ft.EntitySet(id = 'customers')
# Create an entity from the customers dataframe



es = es.entity_from_dataframe(entity_id = 'customers', dataframe = customers_df, 

                              index = 'customer_id', time_index = 'join_date' ,variable_types =  {"zip_code": ft.variable_types.ZIPCode})
es
es = es.entity_from_dataframe(entity_id="transactions",

                                 dataframe=transactions_df,

                                 index="transaction_id",

                               time_index="transaction_time",

                               variable_types={"product_id": ft.variable_types.Categorical})
ft.variable_types.ALL_VARIABLE_TYPES
es
es = es.entity_from_dataframe(entity_id="sessions",

            dataframe=sessions_df,

            index="session_id", time_index = 'session_start')
es




cust_relationship = ft.Relationship(es["customers"]["customer_id"],

                       es["sessions"]["customer_id"])



# Add the relationship to the entity set

es = es.add_relationship(cust_relationship)



sess_relationship = ft.Relationship(es["sessions"]["session_id"],

                       es["transactions"]["session_id"])



# Add the relationship to the entity set

es = es.add_relationship(sess_relationship)



es
feature_matrix, feature_defs = ft.dfs(entityset=es,

                                        target_entity="customers",max_depth = 3)
feature_matrix
len(feature_defs)
feature_defs
# Lets talk about categorical features 

sessions_df.head()

pd.get_dummies(sessions_df['device'],drop_first=True).head()
df = pd.DataFrame(

       [[ 'low', 'London'], [ 'medium', 'New York'], [ 'high', 'Dubai']],

       columns=['Temperature', 'City'])

df
map_dict = {'low':0,'medium':1,'high':2}

def map_values(x):

    return map_dict[x]

df['Temperature_oe'] = df['Temperature'].apply(lambda x: map_values(x))
df
from sklearn.preprocessing import LabelEncoder

# create a labelencoder object

le = LabelEncoder()

# fit and transform on the data

sessions_df['device_le'] = le.fit_transform(sessions_df['device'])

sessions_df.head()
sessions_df.head()


players = pd.read_csv("../input/fifa19/data.csv")
len(players.Club.unique())


from category_encoders.binary import BinaryEncoder

# create a Binaryencoder object

be = BinaryEncoder(cols = ['Club'],)

# fit and transform on the data

players = be.fit_transform(players)
players.head()


players = pd.read_csv("../input/fifa19/data.csv")



from category_encoders.hashing import HashingEncoder

# create a HashingEncoder object

be = HashingEncoder(cols = ['Club'])

# fit and transform on the data

players = be.fit_transform(players)
players.head()
train = pd.read_csv("../input/titanic/train.csv")
train.head()
# taken from https://medium.com/@pouryaayria/k-fold-target-encoding-dfe9a594874b

from sklearn import base

from sklearn.model_selection import KFold



class KFoldTargetEncoderTrain(base.BaseEstimator,

                               base.TransformerMixin):

    def __init__(self,colnames,targetName,

                  n_fold=5, verbosity=True,

                  discardOriginal_col=False):

        self.colnames = colnames

        self.targetName = targetName

        self.n_fold = n_fold

        self.verbosity = verbosity

        self.discardOriginal_col = discardOriginal_col

    def fit(self, X, y=None):

        return self

    def transform(self,X):

        assert(type(self.targetName) == str)

        assert(type(self.colnames) == str)

        assert(self.colnames in X.columns)

        assert(self.targetName in X.columns)

        mean_of_target = X[self.targetName].mean()

        kf = KFold(n_splits = self.n_fold,

                   shuffle = True, random_state=2019)

        col_mean_name = self.colnames + '_' + 'Kfold_Target_Enc'

        X[col_mean_name] = np.nan

        for tr_ind, val_ind in kf.split(X):

            X_tr, X_val = X.iloc[tr_ind], X.iloc[val_ind]

            X.loc[X.index[val_ind], col_mean_name] = X_val[self.colnames].map(X_tr.groupby(self.colnames)

                                     [self.targetName].mean())

            X[col_mean_name].fillna(mean_of_target, inplace = True)

        if self.verbosity:

            encoded_feature = X[col_mean_name].values

            print('Correlation between the new feature, {} and, {} is {}.'.format(col_mean_name,self.targetName,                    

                   np.corrcoef(X[self.targetName].values,

                               encoded_feature)[0][1]))

        if self.discardOriginal_col:

            X = X.drop(self.targetName, axis=1)

        return X
targetc = KFoldTargetEncoderTrain('Pclass','Survived',n_fold=5)

new_train = targetc.fit_transform(train)
new_train[['Pclass_Kfold_Target_Enc','Pclass']].head()
train = pd.read_csv("../input/nyc-taxi-trip-duration/train.csv")
train = train.sample(500)
def haversine_array(lat1, lng1, lat2, lng2): 

    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2)) 

    AVG_EARTH_RADIUS = 6371 # in km 

    lat = lat2 - lat1 

    lng = lng2 - lng1 

    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) *      np.sin(lng * 0.5) ** 2 

    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d)) 

    return h
train['haversine_distance'] = train.apply(lambda x: haversine_array(x['pickup_latitude'], x['pickup_longitude'], x['dropoff_latitude'], x['dropoff_longitude']),axis=1)
def dummy_manhattan_distance(lat1, lng1, lat2, lng2): 

    a = haversine_array(lat1, lng1, lat1, lng2) 

    b = haversine_array(lat1, lng1, lat2, lng1) 

    return a + b
train['manhattan_distance'] = train.apply(lambda x: dummy_manhattan_distance(x['pickup_latitude'], x['pickup_longitude'], x['dropoff_latitude'], x['dropoff_longitude']),axis=1)
def bearing_array(lat1, lng1, lat2, lng2): 

    AVG_EARTH_RADIUS = 6371 # in km 

    lng_delta_rad = np.radians(lng2 - lng1) 

    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2)) 

    y = np.sin(lng_delta_rad) * np.cos(lat2) 

    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad) 

    return np.degrees(np.arctan2(y, x))
train['bearing'] = train.apply(lambda x: bearing_array(x['pickup_latitude'], x['pickup_longitude'], x['dropoff_latitude'], x['dropoff_longitude']),axis=1)
train.loc[:, 'center_latitude'] = (train['pickup_latitude'].values + train['dropoff_latitude'].values) / 2 

train.loc[:, 'center_longitude'] = (train['pickup_longitude'].values + train['dropoff_longitude'].values) / 2
train.head()
import plotly_express as px

px.histogram(train,x='trip_duration')
train['log_trip_duration'] = train['trip_duration'].apply(lambda x: np.log(1+x))
px.histogram(train,x='log_trip_duration')