import numpy as np # mathematical library including linear algebra

import pandas as pd #data processing and CSV file input / output

from sklearn import model_selection, preprocessing # sklearn is the machine learning library

import xgboost as xgb # this is the extreme gradient boosting library
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

macro = pd.read_csv('../input/macro.csv')

id_test = test.id
y_train = train["price_doc"] * .969 + 10

x_train = train.drop(["id", "timestamp", "price_doc"], axis=1)

x_test = test.drop(["id", "timestamp"], axis=1)
for c in x_train.columns:

    if x_train[c].dtype == 'object':

        lbl = preprocessing.LabelEncoder() # set an instance of the label encoder

        lbl.fit(list(x_train[c].values)) # fit it to the values of the training set column headers

        x_train[c] = lbl.transform(list(x_train[c].values)) # Have them transformed to encoded labels

        

for c in x_test.columns:

    if x_test[c].dtype == 'object':

        lbl = preprocessing.LabelEncoder() # set an instance of the label encoder

        lbl.fit(list(x_test[c].values)) # fit it to the values of the test set column headers

        x_test[c] = lbl.transform(list(x_test[c].values)) # Have them transformed to encoded labels
xgb_params = {

    'eta': 0.05,

    'max_depth': 5,

    'subsample': 0.7,

    'colsample_bytree': 0.7,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'silent': 1

}
dtrain = xgb.DMatrix(x_train, y_train)

dtest = xgb.DMatrix(x_test)
cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=200, early_stopping_rounds=20,

    verbose_eval=50, show_stdv=False)
num_boost_rounds = len(cv_output)

model = xgb.train(dict(xgb_params), dtrain, num_boost_round= num_boost_rounds)
y_predict = model.predict(dtest)

output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})



output.to_csv('xgbSub.csv', index=False)