# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV


#######################

## load pima indians dataset
train = pd.read_csv("../input/train.csv")
test =  pd.read_csv("../input/test.csv")

id = test['Id']
test.drop('Id', axis = 1,inplace  = True)
train.drop('Id', axis = 1,inplace  = True)

y = train['SalePrice']

X = train.drop(['SalePrice'], axis=1)

colm = X.select_dtypes(include=['object'])
col = colm.columns


# Create dummy variables for each level of `col`
train_animal_dummies = pd.get_dummies(X[col], prefix=col)
X = X.join(train_animal_dummies)

test_animal_dummies = pd.get_dummies(test[col], prefix=col)
test = test.join(test_animal_dummies)

# Find the difference in columns between the two datasets
# This will work in trivial case, but if you want to limit to just one feature
# use this: f = lambda c: col in c; feature_difference = set(filter(f, train)) - set(filter(f, test))
feature_difference = set(X) - set(test)

# create zero-filled matrix where the rows are equal to the number
# of row in `test` and columns equal the number of categories missing (i.e. set difference 
# between relevant `train` and `test` columns
feature_difference_df = pd.DataFrame(data=np.zeros((test.shape[0], len(feature_difference))),
                                     columns=list(feature_difference))

# add "missing" features back to `test
test = test.join(feature_difference_df)

test  = test.select_dtypes(exclude=['object'])
X  = X.select_dtypes(exclude=['object'])

c = X.columns
test = test[c]




X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.3, random_state=42)

"""
# default objective : regression
# default eta : 0.3
# max_depth [default=6]
parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'learning_rate': [0.2], #so called `eta` value
              'max_depth': [6],
              'min_child_weight': [11],
              'silent': [1],
              'subsample': [0.8],
              'colsample_bytree': [0.7],
              'n_estimators': [500,5000], #number of trees, change it to 1000 for better results
              'missing':[-999],
              'seed': [1337]}
"""

# A parameter grid for XGBoost
params = {'min_child_weight':[3,6,11], 'gamma':[i/10.0 for i in range(2,4)],  'subsample':[i/10.0 for i in range(8,10)],
'colsample_bytree':[i/10.0 for i in range(6,8)], 'max_depth': [4,5,6,7], 'n_estimators': [5000]}

# Initialize XGB and GridSearch
model = xgb.XGBRegressor()

# cv = None, default 3 fold cross validation
grid = RandomizedSearchCV(model, params, cv = 3 , verbose = 10)
grid.fit(X_train, y_train)


# predict

y_pred = grid.best_estimator_.predict(X_test)


from sklearn.metrics import mean_squared_log_error
rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
print("msle is ==== ",rmsle)

#### final prediction #################

final_outcome = grid.best_estimator_.predict(test)

final_outcome = pd.DataFrame(final_outcome)
final_outcome.columns = ['SalePrice']

id = pd.DataFrame(id)
id = id.reset_index()
id = id.drop('index',axis =1)

final_outcome = pd.concat([id,final_outcome],axis = 1)
final_outcome.to_csv("submission3.csv",index = False)
