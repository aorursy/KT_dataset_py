## Building a basic XGBoost to predict success/failure of a project



# load libraries

# create outcome variable

# build modela and inspect feature importances



print("Let's do this...")
# load libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedShuffleSplit

import xgboost as xgb

from xgboost import XGBClassifier



# today, I don't want to see warnings...

import warnings  

warnings.filterwarnings('ignore')



# read all the data

nRowsRead = 10000

data = pd.read_csv('/kaggle/input/archived-kickstarter-projects/kicktraq_kickstarter.csv', delimiter=',', nrows = nRowsRead)

nRow, nCol = data.shape

print(f'There are {nRow} rows and {nCol} columns')
# check for NA

data.isnull().sum()
# create a success flag based on status

data['success'] = [1 if x == "Funding Successful" else 0 for x in data['status']]



# look at proportion of success to failure

data.groupby(['success']).count()
3835 / (4193 + 3835)
# let's build



# create the outcome varible and feature space

y = data.success

X = data.drop(['success','avg_pledge_amount_per_backer','avg_backers_per_pledge_tier','funding_percentage','funding_raised'], axis=1).select_dtypes(exclude=['object'])



# split dataset into training and testing

train_X, test_X, train_y, test_y = train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.25)
# train model

my_model = XGBClassifier(objective = 'binary:logistic')

my_model.fit(train_X, train_y, verbose=False)



# make predictions and find the MAE (why MAE?, no reason, basically random pick here)

predictions = my_model.predict(test_X)

from sklearn.metrics import mean_absolute_error

print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
# what if I wanted to generalize the accuracy of this model?



# use cross validation...

sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=2)

scores = cross_val_score(my_model, train_X, train_y, cv=sss, scoring = "accuracy")

print("Scores:", scores)

print("Mean:", scores.mean())

print("Standard Deviation:", scores.std())
# show me what's driving the model decisions

importances = pd.DataFrame({'feature': X.columns,'importance':np.round(my_model.feature_importances_,3)})

importances = importances.sort_values('importance',ascending=False).set_index('feature')

importances.plot.bar(figsize=(10,5))
my_model_2 = XGBClassifier(objective = 'binary:logistic', n_estimators=1000)

my_model_2.fit(train_X, train_y, early_stopping_rounds=5, 

             eval_set=[(test_X, test_y)], verbose=False)
# make predictions

predictions = my_model_2.predict(test_X)

from sklearn.metrics import mean_absolute_error

print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
my_model_3 = XGBClassifier(objective = 'binary:logistic', n_estimators=1000, learning_rate=0.05)

my_model_3.fit(train_X, train_y, early_stopping_rounds=5, 

             eval_set=[(test_X, test_y)], verbose=False)
# make predictions

predictions = my_model_3.predict(test_X)

from sklearn.metrics import mean_absolute_error

print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
from xgboost import cv



# define data_dmatrix

data_dmatrix = xgb.DMatrix(data=X,label=y)



params = {"objective":"binary:logistic",'colsample_bytree': 0.3,'learning_rate': 0.1,

                'max_depth': 5, 'alpha': 10}



xgb_cv = cv(dtrain=data_dmatrix, params=params, nfold=3,

                    num_boost_round=50, early_stopping_rounds=10, metrics="auc", as_pandas=True, seed=123)
xgb_cv.head()
xgb.plot_importance(my_model_3, max_num_features=10, importance_type='gain')

# There are three methods to measure feature_importances in xgboost.They are :

#    weight : The total number of times this feature was used to split the data across all trees.

#    Cover :The number of times a feature is used to split the data across all trees weighted by the number of training data points that go through those splits.

#    Gain : The average loss reduction gained when using this feature for splitting in trees.
importances = pd.DataFrame({'feature': X.columns,'importance':np.round(my_model_3.feature_importances_,3)})

importances = importances.sort_values('importance',ascending=False).set_index('feature')

importances.plot.bar(figsize=(10,5))