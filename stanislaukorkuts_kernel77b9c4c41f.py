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
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer
train_df = pd.read_csv(r'/kaggle/input/flight-delays-spring-2018/flight_delays_train.csv')

train_df.head()
X_train = train_df.drop('dep_delayed_15min', axis=1)

y_train = train_df['dep_delayed_15min'].map({'Y': 1, 'N': 0})

X_train.head()
for feat in ['Month', 'DayofMonth', 'DayOfWeek']:

    X_train[feat] = train_df[feat].apply(lambda st: int(st.lstrip('c-')))

X_train.head()
from catboost import CatBoostClassifier

ctb = CatBoostClassifier(random_seed=17)

X_train_part, X_valid, y_train_part, y_valid = train_test_split(X_train, y_train, test_size=0.3)

cat_features_idx = np.where(X_train_part.dtypes == 'object')[0].tolist()

ctb.fit(X_train_part, y_train_part, cat_features=cat_features_idx)
roc_auc_score(y_valid, ctb.predict_proba(X_valid)[:, 1])
X_train['Flight'] = X_train['Origin'] + '-' + X_train['Dest']

X_train.drop(['Origin', 'Dest'], axis=1, inplace=True)
for feat in ['UniqueCarrier', 'Flight']:

    X_train = pd.concat([X_train,pd.get_dummies(X_train[feat], prefix=feat)], axis=1)

X_train.drop(['UniqueCarrier', 'Flight'], axis=1, inplace=True)

X_train.head()
X_train.info()
X_train_part, X_valid, y_train_part, y_valid = train_test_split(X_train, y_train, test_size=0.3)
logreg = LogisticRegression()

logreg.fit(X_train_part, y_train_part)
roc_auc_score(y_valid, logreg.predict_proba(X_valid)[:, 1])
xgb = XGBClassifier(n_estimators=30, learning_rate=0.1, seed=17)

parameters = {

             'max_depth':range(3,10,2),

             'min_child_weight':range(1,6,2)

             }

xgb_grid = GridSearchCV(xgb, parameters, scoring='roc_auc', n_jobs=-1)

xgb_grid.fit(X_train_part[['DepTime', 'Distance']], y_train_part)

xgb_grid.best_params_, xgb_grid.best_score_ 
parameters = {

             'max_depth': range(6, 9),

             'min_child_weight': range(2, 5)

             }

xgb_grid = GridSearchCV(xgb, parameters, scoring='roc_auc', n_jobs=-1)

xgb_grid.fit(X_train_part[['DepTime', 'Distance']], y_train_part)

xgb_grid.best_params_, xgb_grid.best_score_ 
xgb = XGBClassifier(n_estimators=30, learning_rate=0.1, max_depth=7, min_child_weight=3, seed=17)
parameters = {

             'gamma': np.linspace(0, 1, 10)

             }

xgb_grid = GridSearchCV(xgb, parameters, scoring='roc_auc', n_jobs=-1)

xgb_grid.fit(X_train_part[['DepTime', 'Distance']], y_train_part)

xgb_grid.best_params_, xgb_grid.best_score_ 
xgb = XGBClassifier(n_estimators=30, learning_rate=0.1, max_depth=7, min_child_weight=3, gamma=1/3, seed=17)
parameters = {

             'reg_alpha': np.logspace(-5, 5, 10)

             }

xgb_grid = GridSearchCV(xgb, parameters, scoring='roc_auc', n_jobs=-1)

xgb_grid.fit(X_train_part[['DepTime', 'Distance']], y_train_part)

xgb_grid.best_params_, xgb_grid.best_score_ 
parameters = {

             'reg_alpha': np.logspace(0, 2, 10)

             }

xgb_grid = GridSearchCV(xgb, parameters, scoring='roc_auc', n_jobs=-1)

xgb_grid.fit(X_train_part[['DepTime', 'Distance']], y_train_part)

xgb_grid.best_params_, xgb_grid.best_score_ 
xgb = XGBClassifier(n_estimators=30, learning_rate=0.1, max_depth=7, min_child_weight=3, gamma=1/3, reg_alpha=2.78, seed=17)
xgb = XGBClassifier(n_estimators=500, learning_rate=0.01, max_depth=7, min_child_weight=3, gamma=1/3, reg_alpha=2.78, seed=17)
xgb.fit(X_train_part[['DepTime', 'Distance']], y_train_part)
roc_auc_score(y_valid, xgb.predict_proba(X_valid[['DepTime', 'Distance']])[:, 1])
def ensemble_predictions(model1, model2, param, X1, X2):

    if not 0 <= param <= 1:

        raise Exception("Incorrect value")

    return param * model1.predict_proba(X1)[:, 1] + (1 - param) * model2.predict_proba(X2)[:, 1]
best_param = 0

best_auc = 0

for param in np.linspace(0, 0.9, 10):

    auc = roc_auc_score(y_valid, ensemble_predictions(logreg, xgb, param, X_valid, X_valid[['DepTime', 'Distance']]))

    if auc > best_auc:

        best_auc = auc

        best_param = param

best_param, best_auc
best_param = 0

best_auc = 0

for param in np.linspace(0.15, 0.25, 10):

    auc = roc_auc_score(y_valid, ensemble_predictions(logreg, xgb, param, X_valid, X_valid[['DepTime', 'Distance']]))

    if auc > best_auc:

        best_auc = auc

        best_param = param

best_param, best_auc
X_test = pd.read_csv(r'/kaggle/input/flight-delays-spring-2018/flight_delays_test.csv')

for feat in ['Month', 'DayofMonth', 'DayOfWeek']:

    X_test[feat] = train_df[feat].apply(lambda st: int(st.lstrip('c-')))

X_test['Flight'] = X_test['Origin'] + '-' + X_test['Dest']

X_test.drop(['Origin', 'Dest'], axis=1, inplace=True)

for feat in ['UniqueCarrier', 'Flight']:

    X_test = pd.concat([X_test,pd.get_dummies(X_test[feat], prefix=feat)], axis=1)

X_test.drop(['UniqueCarrier', 'Flight'], axis=1, inplace=True)

common_features = X_train.columns & X_test.columns

X_test = X_test[common_features]

X_train = X_train[common_features]
xgb = XGBClassifier(n_estimators=500, learning_rate=0.01, max_depth=7, min_child_weight=3, gamma=1/3, reg_alpha=2.78, seed=17)

logreg = LogisticRegression()

xgb.fit(X_train[['DepTime', 'Distance']], y_train)

logreg.fit(X_train, y_train)
predictions = ensemble_predictions(logreg, xgb, 0.16, X_test, X_test[['DepTime', 'Distance']])
predictions
pd.Series(predictions, 

          name='dep_delayed_15min').to_csv('submission.csv', 

                                           index_label='id', header=True)