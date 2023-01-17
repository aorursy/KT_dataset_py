import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
print(os.listdir("../input/"))
train_df = pd.read_csv('../input/train.csv', nrows=2000000)
train_df['distance'] = np.sqrt(
                            np.abs(train_df['pickup_longitude']-train_df['dropoff_longitude'])**2 +
                            np.abs(train_df['pickup_latitude']-train_df['dropoff_latitude'])**2
                                )
train_df.head()
# Separate year month and day
# SUPER SLOW ACTION
# train_df['year'] = pd.DatetimeIndex(train_df['pickup_datetime']).year
# train_df['month'] = pd.DatetimeIndex(train_df['pickup_datetime']).month
# train_df['day'] = pd.DatetimeIndex(train_df['pickup_datetime']).day
test_df = pd.read_csv('../input/test.csv')
test_df['distance'] = np.sqrt(
                            np.abs(test_df['pickup_longitude']-test_df['dropoff_longitude'])**2 +
                            np.abs(test_df['pickup_latitude']-test_df['dropoff_latitude'])**2
                                )
test_df.head()
# Exclude non-sense data
train_df = train_df[train_df.fare_amount>=0]
train_df = train_df.dropna(how = 'any', axis = 'rows')
test_df = test_df.dropna(how = 'any', axis = 'rows')

# Select just the interesting columns
feat_columns = ['passenger_count', 'distance']
label_column = ['fare_amount']
feat_data = train_df[feat_columns]
label_data= train_df[label_column]
# Define fare categories
label_data['fare_cat'] = np.where(label_data['fare_amount'] >= 100, 3, 
                         np.where(label_data['fare_amount'] >= 50, 2, 
                         np.where(label_data['fare_amount'] >= 20, 1, 0)))
label_data = label_data['fare_cat']

# Flatten columns
label_data = np.ravel(label_data)
# Create DMatrix
D_train = xgb.DMatrix(data=feat_data, silent=1, nthread=-1, label =label_data)

# Train parameters
param = { 'silent' : 1,
       'learning_rate' :  0.6,
       'max_depth': 8,
       'tree_method': 'exact',
       'objective': 'multi:softprob',
       "num_class": 4,
       'eval_metric': 'mlogloss' }
n_rounds = 50
bst = xgb.train(param, D_train, n_rounds)
feat_test_data = test_df[feat_columns]

D_test = xgb.DMatrix(data=feat_test_data, silent=1, nthread=-1)
pred = bst.predict( D_test )
predictions = np.asarray([np.argmax(line) for line in pred])
# accuracy = precision_score(test_labels, predictions, average='micro')
# print("Accuracy: %.2f%%" % (accuracy * 100.0))
df2 = feat_test_data
df2.loc[:,('pred')] = list(predictions)
df2.groupby("pred").agg("count")
xgb.plot_importance(bst)
feat_data = train_df[feat_columns]
label_data= train_df[label_column]

label_data['fare_cat'] = np.where(label_data['fare_amount'] >= 100, 6, 
                         np.where(label_data['fare_amount'] >= 50, 5, 
                         np.where(label_data['fare_amount'] >= 35, 4, 
                         np.where(label_data['fare_amount'] >= 20, 3, 
                         np.where(label_data['fare_amount'] >= 10, 2, 
                         np.where(label_data['fare_amount'] >= 5, 1, 0))))))

label_data = label_data['fare_cat']
# Flatten columns
label_data = np.ravel(label_data)
# Create DMatrix
D_train = xgb.DMatrix(data=feat_data, silent=1, nthread=-1, label =label_data)

# Train parameters
param = { 'silent' : 1,
       'learning_rate' :  0.6,
       'max_depth': 8,
       'tree_method': 'exact',
       'objective': 'multi:softprob',
       "num_class": 7,
       'eval_metric': 'mlogloss' }

n_rounds = 50
bst = xgb.train(param, D_train, n_rounds)
feat_test_data = test_df[feat_columns]

D_test = xgb.DMatrix(data=feat_test_data, silent=1, nthread=-1)
pred = bst.predict( D_test )
predictions = np.asarray([np.argmax(line) for line in pred])
# accuracy = precision_score(test_labels, predictions, average='micro')
# print("Accuracy: %.2f%%" % (accuracy * 100.0))
df2 = feat_test_data

df2.loc[:,('pred')] = list(predictions)
df2.groupby("pred").agg("count")
test_df['pred'] = df2['pred'].values
test_df['fare_amount'] = np.where(test_df['pred'] == 6, 100, 
                         np.where(test_df['pred'] == 5, 75, 
                         np.where(test_df['pred'] == 4, 42, 
                         np.where(test_df['pred'] >= 3, 27, 
                         np.where(test_df['pred'] >= 2, 15, 
                         np.where(test_df['pred'] >= 1, 7.5, 3))))))
submission = test_df[['key', 'fare_amount']]
submission.to_csv('submission.csv', index = False)
