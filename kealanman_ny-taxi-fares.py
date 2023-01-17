import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import lightgbm as lgbm



import matplotlib.pyplot as plt

from haversine import haversine

%matplotlib inline

import gc 

import os

print(os.listdir("../input"))



from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.graph_objs as go

init_notebook_mode(connected=True)



# # Class, for use in pipelines, to select certain columns from a DataFrame and convert to a numpy array

# # From A. Geron: Hands-On Machine Learning with Scikit-Learn & TensorFlow, O'Reilly, 2017

# # Modified by Derek Bridge to allow for casting in the same ways as pandas.DatFrame.astype

# class DataFrameSelector(BaseEstimator, TransformerMixin):

#     def __init__(self, attribute_names, dtype=None):

#         self.attribute_names = attribute_names

#         self.dtype = dtype

#     def fit(self, X, y=None):

#         return self

#     def transform(self, X):

#         X_selected = X[self.attribute_names]

#         if self.dtype:

#             return X_selected.astype(self.dtype).values

#         return X_selected.values

    

# # Class, for use in pipelines, to binarize nominal-valued features (while avoiding the dummy variable trap)

# # By Derek Bridge, 2017

# class FeatureBinarizer(BaseEstimator, TransformerMixin):

#     def __init__(self, features_values):

#         self.features_values = features_values

#         self.num_features = len(features_values)

#         self.labelencodings = [LabelEncoder().fit(feature_values) for feature_values in features_values]

#         self.onehotencoder = OneHotEncoder(sparse=False,n_values=[len(feature_values) for feature_values in features_values])

#         self.last_indexes = np.cumsum([len(feature_values) - 1 for feature_values in self.features_values])

#     def fit(self, X, y=None):

#         for i in range(0, self.num_features):

#             X[:, i] = self.labelencodings[i].transform(X[:, i])

#         return self.onehotencoder.fit(X)

#     def transform(self, X, y=None):

#         for i in range(0, self.num_features):

#             X[:, i] = self.labelencodings[i].transform(X[:, i])

#             onehotencoded = self.onehotencoder.transform(X)

#         return np.delete(onehotencoded, self.last_indexes, axis=1)

#     def fit_transform(self, X, y=None):

#         onehotencoded = self.fit(X).transform(X)

#         return np.delete(onehotencoded, self.last_indexes, axis=1)

#     def get_params(self, deep=True):

#         return {"features_values" : self.features_values}

#     def set_params(self, **parameters):

#         for parameter, value in parameters.items():

#             self.setattr(parameter, value)

#         return self

# fields

fields = ['fare_amount', 'pickup_datetime', 'passenger_count', 'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']



# Use pandas to read our training set

df = pd.read_csv("../input/new-york-city-taxi-fare-prediction/train.csv",

                 skipinitialspace=True, 

                 parse_dates = ['pickup_datetime'],

                 infer_datetime_format = True,

                 usecols=fields, 

                 nrows= 10000000)

df.describe()
df.info()
def calc_haversine(row):

    point1= (row['pickup_latitude'],row['pickup_longitude'])

    point2= (row['dropoff_latitude'],row['dropoff_longitude'])

    return(haversine(point1,point2))



def feature_engineer(df, train=True):

    # split the pickup_datetime into year, month, day, hour

    df['year'] = df['pickup_datetime'].dt.year

    df['month'] = df['pickup_datetime'].dt.month

    df['day'] = df['pickup_datetime'].dt.day

    df['hour'] = df['pickup_datetime'].dt.hour

    df.drop(['pickup_datetime'], axis=1, inplace=True)

    

    # converting and then downcast the new columns to smaller sizes

    df['passenger_count'] = pd.to_numeric(df['passenger_count'], downcast='integer')

    df['pickup_latitude'] = pd.to_numeric(df['pickup_latitude'], downcast='float')

    df['pickup_longitude'] = pd.to_numeric(df['pickup_longitude'], downcast='float')

    df['dropoff_latitude'] = pd.to_numeric(df['dropoff_latitude'], downcast='float')

    df['dropoff_longitude'] = pd.to_numeric(df['dropoff_longitude'], downcast='float')

    

    

    

    if(train):

        df['fare_amount'] = pd.to_numeric(df['fare_amount'], downcast='float')





    

    # calculate haversine distance between pickup and dropoff locations

    df['distance'] = df.apply(calc_haversine, axis=1)



    # downcast to save memory

    df['distance'] = pd.to_numeric(df['distance'], downcast='float')



    # checkout the new df

    df.head()

    

    return df



df = feature_engineer(df)

df
df.info()
def clean_data(df, train=True):

    # remove any negative fares, zero passengers and impossible coordinates

    df = df[ 

        (df.passenger_count >= 1) &

        (df.passenger_count < 8) &

        (df.pickup_latitude > 40.5) &

        (df.pickup_latitude < 41) &

        (df.pickup_longitude > -75) &

        (df.pickup_longitude < -73) &

        (df.dropoff_latitude > 40.5) &

        (df.dropoff_latitude < 41) &

        (df.dropoff_longitude > -75) &

        (df.dropoff_longitude < -73) &

        (df.pickup_latitude != 0) &

        (df.pickup_longitude != 0) &

        (df.dropoff_latitude != 0) &

        (df.dropoff_latitude != 0) &

        (df.distance > 0)

           ]

    if(train):

        df = df[(df.fare_amount > 0)]



    # coordinates should fall within these confines 



    #40.507754, -74.255323 # westernmost point

    #40.739021, -73.700556 # easternmost point

    #40.914862, -73.909555 # northernmost point

    #40.496218, -74.247699 # southernmost point

    return df



df = clean_data(df)
df = df.reset_index(drop=True)

df.describe()
# get the labels

y = df.pop('fare_amount').values

x_train = df.iloc[:, df.columns != 'fare_amount']

y_train = y
# Use pandas to read our training set

test_df = pd.read_csv("../input/new-york-city-taxi-fare-prediction/test.csv",

                      skipinitialspace=True, 

                     parse_dates = ['pickup_datetime'],

                     infer_datetime_format = True)

test_df = feature_engineer(test_df, train=False)

#test_df = clean_data(test_df, train=False)

test_keys = test_df.pop('key').values
test_df.describe()
lgbm_params =  {

    'task': 'train',

    'boosting_type': 'gbdt',

    'objective': 'regression',

    'metric': 'rmse',

    'learning_rate': 0.01,

    'num_leaves': 31,

    'max_depth': -1,

    'bagging_freq': 20,

    'colsample_bytree': 0.8,

    'min_gain_to_split': 0.5,

    'num_iterations': 50000,

    'max_bin': 500

}
x_test = test_df



pred_test_y = np.zeros(x_test.shape[0])



train_set = lgbm.Dataset(x_train, y_train)



model = lgbm.train(lgbm_params, train_set=train_set)



pred_test_y = model.predict(x_test, num_iteration = model.best_iteration)



#len(pred_test_y)
submission = pd.read_csv('../input/sample_submission.csv')

submission['fare_amount'] = pred_test_y

submission.to_csv('lgbm_submission.csv', index=False)

submission.head(20)
p_fields = ['STATION','NAME','DATE','PRCP','SNOW','SNWD','TMAX','TMIN']



# Use pandas to read our precipitation data

p_df = pd.read_csv("../input/ny-precipitation-data/1594710.csv",

                   skipinitialspace=True,

                   infer_datetime_format = True,

                   usecols=p_fields

                  )
p_df = p_df[

    p_df.NAME.str.contains('NY CITY CENTRAL PARK, NY US') |

    p_df.NAME.str.contains('STATEN ISLAND 1.4 SE, NY US') |

    p_df.NAME.str.contains('STATEN ISLAND 4.5 SSE, NY US') |

    p_df.NAME.str.contains('BROOKLYN 3.1 NW, NY US') |

    #p_df.NAME.str.contains('BRONX, NY US') | removed as there are only 4 rows of data 

    p_df.NAME.str.contains('JFK INTERNATIONAL AIRPORT, NY US') |

    p_df.NAME.str.contains('LA GUARDIA AIRPORT, NY US')

]

p_df.fillna(0.0)

p_df.describe()



p_df.info()
n = 0

while n < len(p_df.STATION.unique()):

    print(p_df.STATION.unique()[n], p_df.NAME.unique()[n])

    n = n + 1
p_df = p_df.reset_index(drop=True)
central_park = pd.DataFrame(columns=p_fields, data=p_df[p_df.NAME.str.contains("NY CITY CENTRAL PARK, NY US")])

data = [go.Bar(x=central_park.DATE,

            y=p_df.SNOW)]

iplot(data, filename='jupyter-basic_bar')



la_guardia = pd.DataFrame(columns=p_fields, data=p_df[p_df.NAME.str.contains("LA GUARDIA AIRPORT, NY US")])

data = [go.Bar(x=la_guardia.DATE,

            y=p_df.SNOW)]

iplot(data, filename='jupyter-basic_bar')



staten_island = pd.DataFrame(columns=p_fields, data=p_df[p_df.NAME.str.contains("STATEN ISLAND 1.4 SE, NY US")])

data = [go.Bar(x=staten_island.DATE,

            y=p_df.SNOW)]

iplot(data, filename='jupyter-basic_bar')

jfk = pd.DataFrame(columns=p_fields, data=p_df[p_df.NAME.str.contains("JFK INTERNATIONAL AIRPORT, NY US")])

data = [go.Bar(x=jfk.DATE,

            y=p_df.SNOW)]

iplot(data, filename='jupyter-basic_bar')