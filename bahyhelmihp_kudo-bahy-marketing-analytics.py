# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
user_df = pd.read_csv("/kaggle/input/open-shopee-code-league-marketing-analytics/users.csv")

user_df.head()
user_df.shape
train_df = pd.read_csv("/kaggle/input/open-shopee-code-league-marketing-analytics/train.csv")

train_df.head()
train_df.shape
join_df = train_df.merge(user_df, on='user_id', how='inner')

join_df.head()
join_df.shape
for col in join_df:

    print("%s: %s" % (col, join_df[col].dtype))
## Checking max value of these columns
join_df[join_df['last_open_day'] != 'Never open']['last_open_day'].astype('int64').max()
join_df[join_df['last_login_day'] != 'Never login']['last_login_day'].astype('int64').max()
join_df[join_df['last_checkout_day'] != 'Never checkout']['last_checkout_day'].astype('int64').max()
### Drop NA and non-int rows, transform domain columns to categorical
join_df = join_df[join_df['last_open_day'] != 'Never open']

join_df['last_open_day'] = join_df['last_open_day'].astype('int64')



join_df = join_df[join_df['last_login_day'] != 'Never login']

join_df['last_login_day'] = join_df['last_login_day'].astype('int64')



join_df = join_df[join_df['last_checkout_day'] != 'Never checkout']

join_df['last_checkout_day'] = join_df['last_checkout_day'].astype('int64')



join_df = join_df.dropna()

join_df.head()
join_df['domain'].unique()
dict_domain = {}

for i in range(len(join_df['domain'].unique())):

    dict_domain[join_df['domain'].unique()[i]] = i

dict_domain
join_df['domain'] = join_df['domain'].apply(lambda x: dict_domain[x])
join_df.shape
from sklearn.model_selection import cross_val_score, train_test_split

import xgboost

from sklearn.ensemble import RandomForestClassifier
## Also drop columns from users table because test data does not have it

X = join_df.drop(['user_id', 'grass_date', 'open_flag', 'row_id', 'attr_1', 'attr_2', 'attr_3', 'age', 'domain'], axis=1)

y = join_df['open_flag']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=30, random_state=21)
# model = xgboost.XGBClassifier()

clf = RandomForestClassifier()

clf = cross_val_score(clf, X, y, cv=5, scoring='roc_auc')
clf.mean()
from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 10, stop = 500, num = 10)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(1, 10, num = 10)]

max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree

bootstrap = [True, False]

# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}

print(random_grid)
# Use the random grid to search for best hyperparameters

# First create the base model to tune

rf = RandomForestClassifier()

# Random search of parameters, using 3 fold cross validation, 

# search across 100 different combinations, and use all available cores

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 5, cv = 3, verbose=5, random_state=42, n_jobs = -1)

# Fit the random search model

rf_random.fit(X_train, y_train)
rf_random.best_params_
from sklearn.metrics import roc_auc_score



def evaluate(model, test_features, test_labels):

    predictions = model.predict_proba(test_features)[:,1]

    score = roc_auc_score(test_labels, predictions)

    print('Model Performance')

    print('AUC = {:0.2f}'.format(score))

    

    return score
base_model = RandomForestClassifier()

base_model.fit(X_train, y_train)

base_auc = evaluate(base_model, X_test, y_test)
best_random = rf_random.best_estimator_

random_auc = evaluate(best_random, X_test, y_test)
print('Improvement of {:0.2f}%.'.format( 100 * (random_auc - base_auc) / base_auc))
# X = join_df.drop(['user_id', 'grass_date', 'open_flag', 'row_id', 'attr_1', 'attr_2', 'attr_3', 'age', 'domain'], axis=1)

# y = join_df['open_flag']
# # model = xgboost.XGBClassifier()

# model = RandomForestClassifier()

# model.fit(X, y)

model = rf_random.best_estimator_
test_df = pd.read_csv("/kaggle/input/open-shopee-code-league-marketing-analytics/test.csv")

test_df.head()
test_df.shape
for col in test_df:

    print("%s: %s" % (col, test_df[col].dtype))
maxval = test_df[test_df['last_open_day'] != 'Never open']['last_open_day'].astype('int64').max()

test_df['last_open_day'] = test_df['last_open_day'].apply(lambda x: maxval if x == 'Never open' else x)

test_df['last_open_day'] = test_df['last_open_day'].astype('int64') 
maxval = test_df[test_df['last_login_day'] != 'Never login']['last_login_day'].astype('int64').max()

test_df['last_login_day'] = test_df['last_login_day'].apply(lambda x: maxval if x == 'Never login' else x)

test_df['last_login_day'] = test_df['last_login_day'].astype('int64') 
maxval = test_df[test_df['last_checkout_day'] != 'Never checkout']['last_checkout_day'].astype('int64').max()

test_df['last_checkout_day'] = test_df['last_checkout_day'].apply(lambda x: maxval if x == 'Never checkout' else x)

test_df['last_checkout_day'] = test_df['last_checkout_day'].astype('int64') 
X_test = test_df.drop(['user_id', 'grass_date', 'row_id'], axis=1)

X_test.head()
X_test.shape
y_pred = model.predict(X_test)

y_pred
submission_df = pd.read_csv("/kaggle/input/open-shopee-code-league-marketing-analytics/sample_submission_0_1.csv")

submission_df.head()
submission_df['open_flag'] = y_pred

submission_df.head()
submission_df.to_csv("submission_naive_approach_rf.csv", index=False)