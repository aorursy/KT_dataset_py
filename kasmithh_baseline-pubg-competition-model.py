#import necessary packages
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
#load in data
train = pd.read_csv('../input/train_V2.csv')
train = train.dropna()
test = pd.read_csv('../input/test_V2.csv')
Submission = pd.read_csv('../input/sample_submission_V2.csv')
Submission['Id'] = Submission['Id'].astype(str)
#split into X and Y groups
X = train[['weaponsAcquired', 'walkDistance', 'killPlace', 'boosts']]
Y = train['winPlacePerc']
#create X_test for submission later
X_test = test[['weaponsAcquired', 'walkDistance', 'killPlace', 'boosts']]
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.30, random_state=123)
#establish decision tree and get predictions
Tree = DecisionTreeRegressor()
Tree.fit(X_train, Y_train)
Tree_preds = Tree.predict(X_val)
#establish decision tree and get predictions
Forest = RandomForestRegressor(n_estimators = 50, n_jobs = 3)
Forest.fit(X_train,Y_train)
Forest_preds = Forest.predict(X_val)
print("Decision Tree MAE: {}".format(mean_absolute_error(Y_val, Tree_preds)))
print("Random Forest MAE: {}".format(mean_absolute_error(Y_val, Forest_preds)))
#predict X_test
Submission['winPlacePerc'] = Forest.predict(X_test)

#create submission file
Submission.to_csv("Submission.csv", index = False)