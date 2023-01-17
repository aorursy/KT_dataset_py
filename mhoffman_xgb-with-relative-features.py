import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn import model_selection, preprocessing

import xgboost as xgb

import datetime

#now = datetime.datetime.now()



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

macro = pd.read_csv('../input/macro.csv')

id_test = test.id

train.sample(3)
y_train = train["price_doc"]

y_train = pd.Series(np.log1p(y_train.values))

x_train = train.drop(["id", "timestamp", "price_doc"], axis=1)

x_test = test.drop(["id", "timestamp"], axis=1)







#can't merge train with test because the kernel run for very long time



for c in x_train.columns:

    if x_train[c].dtype == 'object':

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(x_train[c].values)) 

        x_train[c] = lbl.transform(list(x_train[c].values))

        #x_train.drop(c,axis=1,inplace=True)

        

for c in x_test.columns:

    if x_test[c].dtype == 'object':

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(x_test[c].values)) 

        x_test[c] = lbl.transform(list(x_test[c].values))

        #x_test.drop(c,axis=1,inplace=True)        
x_train['max_floor'] = x_train['max_floor'].replace(to_replace=0, value=np.nan)

x_train['rel_floor'] = x_train['floor'] / x_train['max_floor'].astype(float)

x_train['rel_kitch_sq'] = x_train['kitch_sq'] / x_train['full_sq'].astype(float)

x_train['rel_life_sq'] = x_train['life_sq'] / x_train['full_sq'].astype(float)

# Corrects for property with zero full_sq.

x_train['rel_life_sq'] = x_train['rel_life_sq'].replace(to_replace=np.inf, value=np.nan)

# Does not account for living room, but reasonable enough.

x_train['avg_room_sq'] = x_train['life_sq'] / x_train['num_room'].astype(float)

# Corrects for studios (zero rooms listed).

x_train['avg_room_sq'] = x_train['avg_room_sq'].replace(to_replace=np.inf, value=np.nan)



x_test['max_floor'] = x_test['max_floor'].replace(to_replace=0, value=np.nan)

x_test['rel_floor'] = x_test['floor'] / x_test['max_floor'].astype(float)

x_test['rel_kitch_sq'] = x_test['kitch_sq'] / x_test['full_sq'].astype(float)

x_test['rel_life_sq'] = x_test['life_sq'] / x_test['full_sq'].astype(float)

# Corrects for property with zero full_sq.

x_test['rel_life_sq'] = x_test['rel_life_sq'].replace(to_replace=np.inf, value=np.nan)

# Does not account for living room, but reasonable enough.

x_test['avg_room_sq'] = x_test['life_sq'] / x_test['num_room'].astype(float)

# Corrects for studios (zero rooms listed).

x_test['avg_room_sq'] = x_test['avg_room_sq'].replace(to_replace=np.inf, value=np.nan)
xgb_params = {

    'eta': 0.04,

    'max_depth': 5,

    'subsample': 0.7,

    'colsample_bytree': 0.7,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'silent': 1

}



dtrain = xgb.DMatrix(x_train, y_train)

dtest = xgb.DMatrix(x_test)
cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,

    verbose_eval=50, show_stdv=False)

cv_output[['train-rmse-mean', 'test-rmse-mean']].plot()
cv_output
num_boost_rounds = len(cv_output)

model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round= num_boost_rounds)
fig, ax = plt.subplots(1, 1, figsize=(8, 13))

xgb.plot_importance(model, max_num_features=50, height=0.5, ax=ax)
y_predict = model.predict(dtest)

y_predict = np.expm1(y_predict)

output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})

output.head()
output.to_csv('xgbSub_2.csv', index=False)