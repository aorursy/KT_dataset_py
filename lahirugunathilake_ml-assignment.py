# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score

from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

import xgboost

from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import tensorflow as tf
from keras import backend as K

from sklearn.model_selection import GridSearchCV

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('../input/my-sample-data/train.csv')
test = pd.read_csv('../input/my-sample-data/test.csv')
train.head()
test.head()
predictor_cols = ['additional_fare', 'duration', 'meter_waiting', 'meter_waiting_fare', 'meter_waiting_till_pickup', 'fare']
train_X = train[predictor_cols]
test_X = test[predictor_cols]

#train_y.fillna(train_y.mean(), inplace=True)
train.fillna(train_X.mean(), inplace=True)
test.fillna(test_X.mean(), inplace=True)
train['num_label'] = False
train['direct_distance'] = 0.0
test['direct_distance'] = 0.0

train['pickup_time'] = pd.to_datetime(train['pickup_time'], errors='coerce')
test['pickup_time'] = pd.to_datetime(test['pickup_time'], errors='coerce')

train['drop_time'] = pd.to_datetime(train['drop_time'], errors='coerce')
test['drop_time'] = pd.to_datetime(test['drop_time'], errors='coerce')

train['pick_hour'] = train['pickup_time'].dt.hour
train['drop_hour'] = train['drop_time'].dt.hour

test['pick_hour'] = test['pickup_time'].dt.hour
test['drop_hour'] = test['drop_time'].dt.hour


for index, row in train.iterrows():
    lat1, lng1, lat2, lng2 = map(np.radians, (row['drop_lat'], row['drop_lon'], row['pick_lat'], row['pick_lon']))
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * 6371 * np.arcsin(np.sqrt(d))
    train.at[index,'direct_distance'] = 2 * 6371 * np.arcsin(np.sqrt(d))
    #d = (((row['drop_lon'] - row['pick_lon'])**2)+((row['drop_lat'] - row['pick_lat'])**2))
    #train.at[index,'direct_distance'] = 2 * 6371 * np.arcsin()
    
for index, row in test.iterrows():
    lat1, lng1, lat2, lng2 = map(np.radians, (row['drop_lat'], row['drop_lon'], row['pick_lat'], row['pick_lon']))
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * 6371 * np.arcsin(np.sqrt(d))
    test.at[index,'direct_distance'] = 2 * 6371 * np.arcsin(np.sqrt(d))


train['fare_for_duration'] = (train['fare'] - train['meter_waiting_fare'])/train['duration']
#train['fare_for_distance'] = (train['fare'] - train['meter_waiting_fare'])/train['direct_distance']
train['fare_for_time'] = (train['fare'] - train['meter_waiting_fare'])/(train['duration']+train['meter_waiting_till_pickup'])
#train['distance_for_time'] = train['direct_distance']/train['duration']
train['meter_waiting_to_duration'] = train['meter_waiting']/train['duration']
train['additional_fare_to_duration'] = train['additional_fare']/train['duration']
#train['additional_fare_to_distance'] = train['additional_fare']/train['direct_distance']
train['additional_fare_to_fare'] = train['additional_fare']/(train['fare']+train['additional_fare'])
train['mtr_wating_fare_to_waiting_duration'] = train['meter_waiting_fare']/(train['meter_waiting']+train['meter_waiting_till_pickup'])
train['additional_fare_to_full_time'] = train['additional_fare']/(train['meter_waiting']+train['meter_waiting_till_pickup']+train['duration'])
train['full_waiting_fare_to_full_time'] = (train['meter_waiting_fare']+train['additional_fare'])/(train['meter_waiting']+train['meter_waiting_till_pickup']+train['duration'])
train['net_fare_for_durtion'] = (train['fare'] - (train['additional_fare']+train['meter_waiting_fare']))/train['duration']

test['fare_for_duration'] = (test['fare'] - test['meter_waiting_fare'])/test['duration']
#test['fare_for_distance'] = (test['fare'] - test['meter_waiting_fare'])/test['direct_distance']
test['fare_for_time'] = (test['fare'] - test['meter_waiting_fare'])/(test['duration']+test['meter_waiting_till_pickup'])
#test['distance_for_time'] = test['direct_distance']/test['duration']
test['meter_waiting_to_duration'] = test['meter_waiting']/test['duration']
test['additional_fare_to_duration'] = test['additional_fare']/test['duration']
#test['additional_fare_to_distance'] = test['additional_fare']/test['direct_distance']
test['additional_fare_to_fare'] = test['additional_fare']/(test['fare']+test['additional_fare'])
test['mtr_wating_fare_to_waiting_duration'] = test['meter_waiting_fare']/(test['meter_waiting']+test['meter_waiting_till_pickup'])
test['additional_fare_to_full_time'] = test['additional_fare']/(test['meter_waiting']+test['meter_waiting_till_pickup']+test['duration'])
test['full_waiting_fare_to_full_time'] = (test['meter_waiting_fare']+test['additional_fare'])/(test['meter_waiting']+test['meter_waiting_till_pickup']+test['duration'])
test['net_fare_for_durtion'] = (test['fare'] - (test['additional_fare']+test['meter_waiting_fare']))/test['duration']

train.replace([np.inf, -np.inf], np.nan)
test.replace([np.inf, -np.inf], np.nan)
train.fillna(train.mean(), inplace=True)
test.fillna(test.mean(), inplace=True)

train['fare_for_duration'] = train['fare_for_duration'].astype(np.float32)
#train['fare_for_distance'] = train['fare_for_distance'].astype(np.float32)
train['fare_for_time'] = train['fare_for_time'].astype(np.float32)
#train['distance_for_time'] = train['distance_for_time'].astype(np.float32)
train['meter_waiting_to_duration'] = train['meter_waiting_to_duration'].astype(np.float32)
train['additional_fare_to_duration'] = train['additional_fare_to_duration'].astype(np.float32)
#train['additional_fare_to_distance'] = train['additional_fare_to_distance'].astype(np.float32)
train['additional_fare_to_fare'] = train['additional_fare_to_fare'].astype(np.float32)
train['mtr_wating_fare_to_waiting_duration'] = train['mtr_wating_fare_to_waiting_duration'].astype(np.float32)
train['additional_fare_to_full_time'] = train['additional_fare_to_full_time'].astype(np.float32)
train['full_waiting_fare_to_full_time'] = train['full_waiting_fare_to_full_time'].astype(np.float32)
train['net_fare_for_durtion'] = train['net_fare_for_durtion'].astype(np.float32)

test['fare_for_duration'] = test['fare_for_duration'].astype(np.float32)
#test['fare_for_distance'] = test['fare_for_distance'].astype(np.float32)
test['fare_for_time'] = test['fare_for_time'].astype(np.float32)
#test['distance_for_time'] = test['distance_for_time'].astype(np.float32)
test['meter_waiting_to_duration'] = test['meter_waiting_to_duration'].astype(np.float32)
test['additional_fare_to_duration'] = test['additional_fare_to_duration'].astype(np.float32)
#test['additional_fare_to_distance'] = test['additional_fare_to_distance'].astype(np.float32)
test['additional_fare_to_fare'] = test['additional_fare_to_fare'].astype(np.float32)
test['mtr_wating_fare_to_waiting_duration'] = test['mtr_wating_fare_to_waiting_duration'].astype(np.float32)
test['additional_fare_to_full_time'] = test['additional_fare_to_full_time'].astype(np.float32)
test['full_waiting_fare_to_full_time'] = test['full_waiting_fare_to_full_time'].astype(np.float32)
test['net_fare_for_durtion'] = test['net_fare_for_durtion'].astype(np.float32)

for index, row in train.iterrows():
    if(row['label'] == "correct"):
        train.at[index, 'num_label'] = 1
    else:
        train.at[index, 'num_label'] = 0


#train_y = train['label']
# 'fare_for_distance',
# 'distance_for_time', 'additional_fare_to_distance', 'distance_for_time', 'only_fare_for_duration',
predictor_cols = ['additional_fare', 'net_fare_for_durtion', 'full_waiting_fare_to_full_time', 'additional_fare_to_full_time', 'mtr_wating_fare_to_waiting_duration', 'fare_for_time', 'meter_waiting_to_duration', 'fare_for_duration', 'additional_fare_to_duration', 'additional_fare_to_fare', 'pick_hour', 'drop_hour', 'pick_lat', 'pick_lon', 'direct_distance', 'duration', 'meter_waiting', 'meter_waiting_fare', 'meter_waiting_till_pickup', 'fare']
train_X_ = train[predictor_cols]
test_X = test[predictor_cols]
train_X_.replace([np.inf, -np.inf], np.nan)
test_X.replace([np.inf, -np.inf], np.nan)
train_X_.fillna(train_X_.mean(), inplace=True)
test_X.fillna(test_X.mean(), inplace=True)

train_y_ = train.num_label
train_X_
test_X
train_X, val_X, train_y, val_y = train_test_split(train_X_, train_y_, test_size = 0.2, train_size = 0.8, random_state = 0)
def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    mse = mean_squared_error(test_labels, predictions)
    mae = mean_absolute_error(test_labels, predictions)
    rmse = np.sqrt(mse)
    score=cross_val_score(model, test_features, test_labels, cv=10)
    accuracy = accuracy_score(test_labels, predictions.round())
    print('Model Performance')
    print('Accuracy:%f'%accuracy)
    print("Mean cross validation score:%f"%score.mean())
    print('Mean Squared Error : %.4f' % mse)
    print('Root MSE : %.4f' % rmse)
my_random_forest_model = RandomForestRegressor(n_estimators=100, random_state = 0)
my_random_forest_model.fit(train_X, train_y)

evaluate(my_random_forest_model, val_X, val_y)

pred_classes = my_random_forest_model.predict(val_X)
predi = []
for x in range(len(pred_classes)):
    predi.append(bool(pred_classes[x]))
    
pred_classes = np.array(predi)

f1 = f1_score(val_y, pred_classes)
print('F1 score: %f' % f1)
# Multiply by -1 since sklearn calculates *negative* MAE
#scores = cross_val_score(my_pipeline, X, y, cv=5, scoring='neg_mean_absolute_error')
#scores = cross_val_score(my_random_forest_model, train_X, train_y.round(), scoring='f1')
#print("MAE scores:\n", scores)

my_decision_tree_model = DecisionTreeRegressor(min_samples_split=100,
        max_features="auto", random_state=50, 
        max_depth=100)
my_decision_tree_model.fit(train_X, train_y)

evaluate(my_decision_tree_model, val_X, val_y)

pred_classes = my_decision_tree_model.predict(val_X)
predi = []
for x in range(len(pred_classes)):
    predi.append(bool(pred_classes[x]))
    
pred_classes = np.array(predi)

f1 = f1_score(val_y, pred_classes)
print('F1 score: %f' % f1)
Ada_boost_model = AdaBoostRegressor()
Ada_boost_model.fit(train_X, train_y)

evaluate(Ada_boost_model, val_X, val_y)

pred_classes = Ada_boost_model.predict(val_X)
predi = []
for x in range(len(pred_classes)):
    predi.append(bool(pred_classes[x]))
    
pred_classes = np.array(predi)

f1 = f1_score(val_y, pred_classes)
print('F1 score: %f' % f1)
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(C=1, penalty='l1', solver='liblinear')
param_grid = {'C':[0.001,.009,0.01,.09,1,5,10,25]}
'''{
    'n_estimators' : [10, 200],
    'max_features' : ['auto', 'sqrt', 'log2', 0.5]
}'''
gs_model = GridSearchCV(clf, param_grid = param_grid, scoring = 'recall')
gs_model.fit(train_X, train_y)

predictions = gs_model.predict(val_X)
accuracy = accuracy_score(val_y, predictions.round())
print('Accuracy:%f'%accuracy)
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
#def create_baseline():
'''
model = Sequential()
model.add(Dense(7, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(7, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(7, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(7, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(7, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(7, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(7, activation='relu'))
#model.add(Dropout(0.4))
model.add(Dense(1, activation='softmax'))
#model.add(Dense(1, activation='sigmoid'))
#model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.compile(loss='mse', optimizer='Adam', metrics=[tf.keras.metrics.MeanSquaredError(),tf.keras.metrics.Accuracy()])
#tf.keras.metrics.BinaryAccuracy()
#adam,sgd
#return model
'''
from keras.layers import BatchNormalization
model = Sequential()
model.add(Dense(15))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(256,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(128,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(64,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(32,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(16,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(8,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1,activation='softmax'))

model.compile(optimizer='adam',loss='mse', metrics=['accuracy',f1_m])
#model.compile(loss=keras.losses.categorical_crossentropy,optimizer='adam',metrics=['accuracy'])

#estimator = KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=10, verbose=0)
#kfold = StratifiedKFold(n_splits=10, shuffle=True)
#results = cross_val_score(estimator, train_X, train_y, cv=kfold)
#print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
model.fit(train_X_.values, train_y_.values,
          batch_size=20,
          epochs=5,
          validation_split = 0.2,
          verbose = 1,
          shuffle=True)

#evaluate(model, val_X, val_y)
pred_classes = model.predict(val_X)
predi = []
for x in range(len(pred_classes)):
    predi.append(bool(pred_classes[x]))
    
pred_classes = np.array(predi)

f1 = f1_score(val_y, pred_classes)
print('F1 score: %f' % f1)
xg_model = xgboost.XGBClassifier(base_score=0.1, booster= None, colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, eval_metric='auc',
              gamma=0.4, gpu_id=0, importance_type='gain',
              interaction_constraints=None, learning_rate=0.01,
              max_delta_step=0, max_depth=80, min_child_weight=7, missing=None,
              monotone_constraints=None, n_estimators=1000, n_jobs=6, nthread=6,
              num_parallel_tree=1, objective='binary:logistic', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, silent=False,
              subsample=0.8,tree_method='hist', validate_parameters=False,
              verbosity=1)
#xg_model = xgboost.XGBClassifier()
#base_score 0.5 -> 0.1
#score=cross_val_score(xg_model, train_X, train_y, cv=10)
xg_model.fit(train_X, train_y)

def xg_evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    #mse = mean_squared_error(test_labels, predictions)
    #mae = mean_absolute_error(test_labels, predictions)
    #rmse = np.sqrt(mse)
    #score=cross_val_score(model, test_features, test_labels, cv=10)
    accuracy = accuracy_score(test_labels, predictions.round())
    #print('Model Performance')
    print('Accuracy:%f'%accuracy)
    #print("Mean cross validation score:%f"%score.mean())
    #print('Mean Squared Error : %.4f' % mse)
    #print('Root MSE : %.4f' % rmse)
xg_evaluate(xg_model, val_X, val_y)
pred_classes = xg_model.predict(val_X)
predi = []
for x in range(len(pred_classes)):
    predi.append(bool(pred_classes[x]))
    
pred_classes = np.array(predi)

f1 = f1_score(val_y, pred_classes)
print('F1 score: %f' % f1)
from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier(n_neighbors=7)
knn_classifier.fit(train_X, train_y)

def knn_evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    #mse = mean_squared_error(test_labels, predictions)
    #mae = mean_absolute_error(test_labels, predictions)
    #rmse = np.sqrt(mse)
    #score=cross_val_score(model, test_features, test_labels, cv=10)
    accuracy = accuracy_score(test_labels, predictions.round())
    #print('Model Performance')
    print('Accuracy:%f'%accuracy)
    #print("Mean cross validation score:%f"%score.mean())
    #print('Mean Squared Error : %.4f' % mse)
    #print('Root MSE : %.4f' % rmse)
knn_evaluate(knn_classifier, val_X, val_y)

pred_classes = knn_classifier.predict(val_X)
predi = []
for x in range(len(pred_classes)):
    predi.append(bool(pred_classes[x]))
    
pred_classes = np.array(predi)

f1 = f1_score(val_y, pred_classes)
print('F1 score: %f' % f1)
predicted_rf = my_random_forest_model.predict(test_X)
predicted_dt = my_decision_tree_model.predict(test_X)
predicted_ad = Ada_boost_model.predict(test_X)
predicted_xg = xg_model.predict(test_X)
predicted_NN = model.predict(test_X)
predicted_KNN = knn_classifier.predict(test_X)

pred_KNN = []
for x in range(len(predicted_KNN)):
    pred_KNN.append(bool(predicted_KNN[x]))

predicted_KNN = np.array(pred_KNN)

pred_NN = []
for x in range(len(predicted_NN)):
    pred_NN.append(bool(predicted_NN[x]))

predicted_NN = np.array(pred_NN)
    
pred_rf = []
for x in range(len(predicted_rf)):
    pred_rf.append(bool(predicted_rf[x]))
    
predicted_rf = np.array(pred_rf)
    
pred_dt = []
for x in range(len(predicted_dt)):
    pred_dt.append(bool(predicted_dt[x]))

predicted_dt = np.array(pred_dt)

pred_ad = []
for x in range(len(predicted_ad)):
    pred_ad.append(bool(predicted_ad[x]))

predicted_ad = np.array(pred_ad)

pred_xg = []
for x in range(len(predicted_xg)):
    pred_xg.append(bool(predicted_xg[x]))
    
predicted_xg = np.array(pred_xg)

output_rf = pd.DataFrame({'tripid': test.tripid,
                       'prediction': predicted_rf})

output_dt = pd.DataFrame({'tripid': test.tripid,
                       'prediction': predicted_dt})

output_adb = pd.DataFrame({'tripid': test.tripid,
                       'prediction': predicted_dt})

output_xg = pd.DataFrame({'tripid': test.tripid,
                       'prediction': predicted_xg})

output_NN = pd.DataFrame({'tripid': test.tripid,
                       'prediction': predicted_NN})

output_KNN = pd.DataFrame({'tripid': test.tripid,
                       'prediction': predicted_KNN})

output_rf_path = "submission_rf.csv"
output_dt_path = "submission_dt.csv"
output_adb_path = "submission_adb.csv"
output_xg_path = "submission_xg.csv"
output_NN_path = "submission_nn.csv"
output_KNN_path = "submission_knn.csv"

output_rf.to_csv(output_rf_path, index=False)
output_dt.to_csv(output_dt_path, index=False)
output_adb.to_csv(output_adb_path, index = False)
output_xg.to_csv(output_xg_path, index = False)
output_NN.to_csv(output_NN_path, index = False)
output_KNN.to_csv(output_KNN_path, index = False)