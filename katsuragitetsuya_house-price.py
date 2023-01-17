import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt

import seaborn as sns

%matplotlib inline



pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)
home = '/kaggle/input/house-prices-advanced-regression-techniques/'



train = pd.read_csv(home + 'train.csv')

test = pd.read_csv(home + 'test.csv')

print('Shape of train data: ' + str(train.shape))

print('Shape of test data: ' + str(test.shape))



test_copy = test.copy()
def display_missing_value(data):

    items = data.isnull().sum()[data.isnull().sum() > 0]

    col_list = data.isnull().sum()[data.isnull().sum() > 0].index.tolist()

    col_type_list = data[col_list].dtypes

    missing_items = pd.concat([items, col_type_list], axis = 1)

    missing_items = missing_items.rename(columns = {0:'count', 1:'type'})

    return missing_items



print('Missing value of train data: \n{}'.format(display_missing_value(train)))
print('Missing value of test data: \n{}'.format(display_missing_value(test)))
def preprocess(data):

    data = data.drop('Id', axis=1)

    for col in data.columns:

        if data[col].dtypes == 'object':

            data[col] = data[col].fillna('NA')

        elif data[col].dtypes in ('int32', 'int64', 'float64'):

            data[col] = data[col].fillna(0)

    

    return data



train = preprocess(train)

test = preprocess(test)
train.head()
fig, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(train.corr(), square=True, vmax=1, vmin=-1, center=0, cmap='YlGnBu')
plt_cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars']

sns.pairplot(train[plt_cols])

plt.show()
obj_cols = set(train.dtypes[train.dtypes == 'object'].index.tolist() + test.dtypes[test.dtypes == 'object'].index.tolist())



for obj_col in obj_cols:

    cat = set(train[obj_col].unique().tolist() + test[obj_col].unique().tolist())

    

    train[obj_col] = pd.Categorical(train[obj_col], categories=cat)

    test[obj_col] = pd.Categorical(test[obj_col], categories=cat)

    

train = pd.get_dummies(train)

test = pd.get_dummies(test)
sns.distplot(train['SalePrice'])

plt.show()

sns.distplot(np.log(train['SalePrice']))

plt.show()

X = train.drop('SalePrice', axis=1)

y = np.log(train['SalePrice'])
from sklearn.linear_model import ElasticNet

from sklearn.metrics import r2_score



parms = [0.001, 0.01, 0.1, 1.0, 10.0,100.0,1000.0]



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)



for parm in parms:

    enet = ElasticNet(alpha=parm)

    enet.fit(X_train, y_train)

    pred = enet.predict(X_test)

    print('parm is {0} : score is {1}'.format(parm, r2_score(y_test, pred)))

    
from sklearn.linear_model import Lasso



for parm in parms:

    lasso = Lasso(alpha=parm)

    lasso.fit(X_train, y_train)

    pred = lasso.predict(X_test)

    print('parm is {0} : score is {1}'.format(parm, r2_score(y_test, pred)))

from sklearn.ensemble import RandomForestRegressor



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 666)



rfr = RandomForestRegressor(100)

rfr = rfr.fit(X_train, y_train)

score = rfr.score(X_test, y_test)

print(score)
import xgboost as xgb

from sklearn.model_selection import GridSearchCV



# reg = xgb.XGBRegressor(n_estimators=10000)



# params = {'max_depth': [2, 4, 6], 'alpha': [0.1, 1, 10]}



# reg_cv = GridSearchCV(reg, params, verbose=1)

# reg_cv.fit(X_train, y_train, early_stopping_rounds=42, eval_set=[[X_test, y_test]])

# print(reg_cv.best_params_, reg_cv.best_score_)

# Best is {'alpha': 0.1, 'max_depth': 4} 0.8897839007072111
reg = xgb.XGBRegressor(alpha=0.1, base_score=0.5, booster='gbtree', colsample_bylevel=1,

             colsample_bynode=1, colsample_bytree=1, gamma=0,

             importance_type='gain', learning_rate=0.1, max_delta_step=0,

             max_depth=4, min_child_weight=1, missing=None, n_estimators=100,

             n_jobs=1, nthread=None, objective='reg:linear', random_state=0,

             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,

             silent=None, subsample=1, verbosity=1)

reg.fit(X_train, y_train)
# enet = ElasticNet(alpha=0.001)

# enet.fit(X_train, y_train)

# pred = enet.predict(X_test)



# pred = rfr.predict(X_train)



pred_train = reg.predict(X_train)

pred_test = reg.predict(X_test)



ax = xgb.plot_importance(reg)

fig = ax.figure

fig.set_size_inches(10, 30)
# test_price = np.exp(enet.predict(test))

pred_sub = reg.predict(test)

test_price = np.exp(pred_sub)
sub = pd.DataFrame(test_copy['Id'])

sub['SalePrice'] = pd.DataFrame(test_price)

sub.to_csv('submission.csv', index=False)