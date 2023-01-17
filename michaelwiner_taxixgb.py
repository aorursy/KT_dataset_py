import numpy as np # linear algebra
import pandas as pd # CSV file I/O (e.g. pd.read_csv)
import os # reading the input files we have access to
import xgboost as xgb
import sklearn as sckit
import matplotlib.pyplot as plt
import calendar

print(os.listdir())
train_df_v1 =  pd.read_csv('../input/train.csv', nrows = 10000000 , parse_dates= ['key'] )
def add_travel_vector_features(df):
    df['abs_diff_longitude'] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df['abs_diff_latitude'] = (df.dropoff_latitude - df.pickup_latitude).abs()

add_travel_vector_features(train_df_v1)
def add_date_data(df):
    train_df_v1['weekday']=train_df_v1.key.apply(lambda x:calendar.day_name[x.weekday()])
    train_df_v1['hour'] = train_df_v1.key.map( lambda x: x.hour )
    train_df_v1['day'] = train_df_v1.key.map( lambda x: x.day )
    train_df_v1['month'] = train_df_v1.key.map( lambda x: x.month )
    train_df_v1['year'] = train_df_v1.key.map( lambda x: x.year )
add_date_data(train_df_v1)
print('Old size: %d' % len(train_df_v1))
train_df_v1 = train_df_v1.dropna(how = 'any', axis = 'rows')
features_v1_list = ['pickup_longitude','pickup_latitude', 'dropoff_longitude',
                     'dropoff_latitude','passenger_count', 'abs_diff_longitude', 'abs_diff_latitude'
                           ,'hour','day','month','year','weekday']

train_df_v1 = train_df_v1.loc[train_df_v1['abs_diff_latitude'] < 1 ]
train_df_v1 = train_df_v1.loc[train_df_v1['abs_diff_longitude'] < 1 ]

train_df_v1 = train_df_v1.loc[train_df_v1['abs_diff_latitude'] > 0.001 ]
train_df_v1 = train_df_v1.loc[train_df_v1['abs_diff_longitude'] > 0.001 ]


train_df_v1 = train_df_v1.loc[train_df_v1['pickup_longitude'] < -72 ]
train_df_v1 = train_df_v1.loc[train_df_v1['pickup_longitude'] > -75 ]


train_df_v1 = train_df_v1.loc[train_df_v1['pickup_latitude'] < 41 ]
train_df_v1 = train_df_v1.loc[train_df_v1['pickup_latitude'] > 39 ]

train_df_v1 = train_df_v1.loc[train_df_v1['fare_amount'] > 1 ]

print('New size: %d' % len(train_df_v1))
train_df_v1_features = train_df_v1[features_v1_list]
train_df_v1.describe()
class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)
from sklearn.preprocessing import LabelEncoder 
train_df_v1_features_new = MultiColumnLabelEncoder(columns = ['weekday']).fit_transform(train_df_v1_features)

def dist(pickup_lat, pickup_long, dropoff_lat, dropoff_long):  
    distance = np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)
    
    return distance
nyc = (-74.0063889, 40.7141667)
jfk = (-73.7822222222, 40.6441666667)
ewr = (-74.175, 40.69)
lgr = (-73.87, 40.77)
train_df_v1_features_new['distance_to_center'] = dist(nyc[1], nyc[0],
                                      train_df_v1_features_new['pickup_latitude'], train_df_v1_features_new['pickup_longitude'])
train_df_v1_features_new['pickup_distance_to_jfk'] = dist(jfk[1], jfk[0],
                                         train_df_v1_features_new['pickup_latitude'], train_df_v1_features_new['pickup_longitude'])
train_df_v1_features_new['dropoff_distance_to_jfk'] = dist(jfk[1], jfk[0],
                                           train_df_v1_features_new['dropoff_latitude'], train_df_v1_features_new['dropoff_longitude'])
train_df_v1_features_new['pickup_distance_to_ewr'] = dist(ewr[1], ewr[0], 
                                          train_df_v1_features_new['pickup_latitude'], train_df_v1_features_new['pickup_longitude'])
train_df_v1_features_new['dropoff_distance_to_ewr'] = dist(ewr[1], ewr[0],
                                           train_df_v1_features_new['dropoff_latitude'], train_df_v1_features_new['dropoff_longitude'])
train_df_v1_features_new['pickup_distance_to_lgr'] = dist(lgr[1], lgr[0],
                                          train_df_v1_features_new['pickup_latitude'], train_df_v1_features_new['pickup_longitude'])
train_df_v1_features_new['dropoff_distance_to_lgr'] = dist(lgr[1], lgr[0],
                                           train_df_v1_features_new['dropoff_latitude'], train_df_v1_features_new['dropoff_longitude'])
label_v1 = train_df_v1['fare_amount']
def XGB_regressor(train_X, train_y, feature_names=None, seed_val=2017, num_rounds=500):
    param = {}
    param['objective'] = 'reg:linear'
    param['eta'] = 0.05
    param['max_depth'] = 8
    param['silent'] = 1
    param['eval_metric'] = 'rmse'
    param['min_child_weight'] = 1
    param['subsample'] = 0.7
    param['colsample_bytree'] = 0.7
    param['seed'] = seed_val
    num_rounds = num_rounds

    plst = list(param.items())

    xgtrain = xgb.DMatrix(train_X, label=train_y)
    model = xgb.train(plst, xgtrain, num_rounds)
        
    return model    
model = XGB_regressor(train_X = train_df_v1_features_new , train_y = label_v1 )
train_df_v1_features_new[0:5]
xgtrain = xgb.DMatrix(train_df_v1_features_new, label=label_v1)
preds_v1 = model.predict (xgtrain)
rmse = np.sqrt(sckit.metrics.mean_squared_error(label_v1, preds_v1))
print("RMSE: %f" % (rmse))
test_df_v1 =  pd.read_csv('../input/test.csv' )
testKey =  pd.read_csv('../input/test.csv' )
test_df_v1_fordates = test_df_v1
test_df_v1_fordates['key']= pd.to_datetime(test_df_v1_fordates['key'])
test_df_v1_fordates['hour'] = test_df_v1_fordates.key.map( lambda x: x.hour )
test_df_v1_fordates['day'] = test_df_v1_fordates.key.map( lambda x: x.day )
test_df_v1_fordates['month'] = test_df_v1_fordates.key.map( lambda x: x.month )
test_df_v1_fordates['year'] = test_df_v1_fordates.key.map( lambda x: x.year )
test_df_v1_fordates['weekday']=test_df_v1_fordates.key.apply(lambda x:calendar.day_name[x.weekday()])
add_travel_vector_features(test_df_v1_fordates)
test_df_v1_features = test_df_v1_fordates[features_v1_list]
test_df_v1_features = MultiColumnLabelEncoder(columns = ['weekday']).fit_transform(test_df_v1_features)
test_df_v1_features['distance_to_center'] = dist(nyc[1], nyc[0],
                                      test_df_v1_features['pickup_latitude'], test_df_v1_features['pickup_longitude'])
test_df_v1_features['pickup_distance_to_jfk'] = dist(jfk[1], jfk[0],
                                         test_df_v1_features['pickup_latitude'], test_df_v1_features['pickup_longitude'])
test_df_v1_features['dropoff_distance_to_jfk'] = dist(jfk[1], jfk[0],
                                           test_df_v1_features['dropoff_latitude'], test_df_v1_features['dropoff_longitude'])
test_df_v1_features['pickup_distance_to_ewr'] = dist(ewr[1], ewr[0], 
                                          test_df_v1_features['pickup_latitude'], test_df_v1_features['pickup_longitude'])
test_df_v1_features['dropoff_distance_to_ewr'] = dist(ewr[1], ewr[0],
                                           test_df_v1_features['dropoff_latitude'], test_df_v1_features['dropoff_longitude'])
test_df_v1_features['pickup_distance_to_lgr'] = dist(lgr[1], lgr[0],
                                          test_df_v1_features['pickup_latitude'], test_df_v1_features['pickup_longitude'])
test_df_v1_features['dropoff_distance_to_lgr'] = dist(lgr[1], lgr[0],
                                           test_df_v1_features['dropoff_latitude'], test_df_v1_features['dropoff_longitude'])
xgtrainTest = xgb.DMatrix(test_df_v1_features)
preds_v1_test = model.predict(xgtrainTest)
submission = pd.DataFrame(
    {'key': testKey.key, 'fare_amount': preds_v1_test },
    columns = ['key', 'fare_amount'])
submission.to_csv('submissionXGlast.csv', index = False)
