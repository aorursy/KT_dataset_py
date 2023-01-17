import datetime

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sn

from sklearn.metrics import mean_squared_log_error

import os



TRAIN_DATA = os.path.join('../input', 'train.csv')

TEST_DATA = os.path.join('../input', 'test.csv')



train_df = pd.read_csv(TRAIN_DATA)

test_df = pd.read_csv(TEST_DATA)

test_datetime = test_df['datetime']

print('Train data set size: {}'.format(train_df.shape))

print('Test data set size: {}'.format(test_df.shape))



print(train_df.dtypes)

print(test_df.dtypes)

print('N/A values in train data: \n{}'.format(train_df.isnull().sum()))



print('Randomly selected 5 rows from the dataframe:')

train_df.sample(frac=1).head(5)

train_df.describe()
i=0

for column in ['season', 'holiday', 'workingday', 'weather']:

    plt.subplot(1, 4, i+1)

    sn.countplot(column, data=train_df)

    i = i+1
# rental hour

train_df['hour'] = train_df['datetime'].apply(lambda x: x.split()[1].split(':')[0]).astype(int)

test_df['hour'] = test_df['datetime'].apply(lambda x: x.split()[1].split(':')[0]).astype(int)

# rental month

train_df['month'] = train_df['datetime'].apply(lambda x: x.split()[0].split('-')[1]).astype(int)

test_df['month'] = test_df['datetime'].apply(lambda x: x.split()[0].split('-')[1]).astype(int)

# day of the week

train_df['weekday'] = train_df['datetime'].apply(lambda x: datetime.datetime.strptime(x.split()[0].split(':')[0], '%Y-%m-%d').strftime('%w')).astype(int)

test_df['weekday'] = test_df['datetime'].apply(lambda x: datetime.datetime.strptime(x.split()[0].split(':')[0], '%Y-%m-%d').strftime('%w')).astype(int)

sn.swarmplot(x='hour', y='temp', hue='season', data=train_df)
sn.heatmap(train_df.corr())
CATEGORICAL_VARS = ['season', 'holiday', 'workingday', 'weather', 'hour', 'month', 'weekday']

for var in CATEGORICAL_VARS:

    train_df[var] = train_df[var].astype('category')
# drop unnecessary columns

train_df = train_df.drop(['datetime', 'casual', 'registered'], axis=1)

test_df = test_df.drop(['datetime'], axis=1)



train_df.sample(frac=1).head(5)



y_train = train_df['count']

X_train = train_df.drop(['count'], axis=1)
from sklearn.tree import DecisionTreeRegressor, export_graphviz



decisionTree = DecisionTreeRegressor()



decisionTree.fit(X = X_train, y = y_train)

pred_tree = decisionTree.predict(X = X_train)



print("RMSLE on training data for a single Decision Tree with default parameters: {:.4f}".format(np.sqrt(mean_squared_log_error(y_train, pred_tree))))

# RMSLSE: 0.01078 - looks like a single Decision Tree with default parameters overfit to the train dataset

print('Probable overfitting')



decisionTree2 = DecisionTreeRegressor(max_depth=5)

decisionTree2.fit(X = X_train, y = y_train)

pred_tree2 = decisionTree2.predict(X = X_train)



print("RMSLE on training data for a single Decision Tree with maximum depth 5: {:.4f}".format(np.sqrt(mean_squared_log_error(y_train, pred_tree2))))
from sklearn.ensemble import RandomForestRegressor

randomForest = RandomForestRegressor(n_estimators=100)



randomForest.fit(X = X_train, y = y_train)

pred_rf = randomForest.predict(X = X_train)



print("RMSLE on training data for Random Forest: {:.4f}".format(np.sqrt(mean_squared_log_error(y_train, pred_rf))))
from sklearn.ensemble import GradientBoostingRegressor

gbm = GradientBoostingRegressor(n_estimators=1000)

gbm.fit(X = X_train, y = np.log(y_train))

pred_gbm = gbm.predict(X = X_train)

print("RMSLE on training data for Gradient Boosting: ", np.sqrt(mean_squared_log_error(y_train, np.exp(pred_gbm))))
predsTest = gbm.predict(X= test_df)

fig,(ax1,ax2)= plt.subplots(ncols=2)

fig.set_size_inches(12,5)

sn.distplot(y_train, ax=ax1, bins=50)

sn.distplot(np.exp(predsTest), ax=ax2, bins=50)
submission_df = pd.DataFrame({

        "datetime": test_datetime,

        "count": [max(0, x) for x in np.exp(predsTest)]

    })

submission_df.to_csv('../submission_gbm.csv', index=False)