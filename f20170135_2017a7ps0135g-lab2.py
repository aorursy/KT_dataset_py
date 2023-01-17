import pandas as pd

import numpy as np

import xgboost as xgb

import os

import seaborn as sns
DATA_DIR = '/kaggle/input/eval-lab-2-f464/'

DATA_TRAIN = 'train.csv'

DATA_TEST = 'test.csv'



df_train = pd.read_csv(os.path.join(DATA_DIR, DATA_TRAIN))



df_train.corr()
# DATA_DIR = '.'

# DATA_TRAIN = 'train.csv'

# DATA_TEST = 'test.csv'



# dtrain_x = xgb.DMatrix(os.path.join(DATA_DIR, DATA_TRAIN+'?format=csv&label_column=10'), silent = True)

# dtest_x = xgb.DMatrix(os.path.join(DATA_DIR, DATA_TEST+'?format=csv&label_column=10'), silent = True)





df_train_data = df_train[['chem_1', 'chem_4', 'chem_5', 'chem_6', 'attribute']] # doesn't include id and class

df_train_label = df_train[[df_train.columns[-1]]] # only class

dtrain = xgb.DMatrix(data = df_train_data, label = df_train_label)



df_test = pd.read_csv(os.path.join(DATA_DIR, DATA_TEST))

df_test_data = df_test[['chem_1', 'chem_4', 'chem_5', 'chem_6', 'attribute']] # doesn't include id

# df_test_label = df_test[[df_test.columns[-1]]] # only class

dtest = xgb.DMatrix(data = df_test_data)
df_train.columns
print(dtrain.num_row(), dtrain.num_col(), dtest.num_row(), dtest.num_col())
n_total = dtrain.num_row()

n_train = (int)(n_total*0.85)

n_eval = n_total - n_train



dtrain_train = dtrain.slice(list(np.arange(n_train)))

dtrain_eval = dtrain.slice(list(np.arange(n_train, n_total, 1)))



print(dtrain_train.num_row(), dtrain_eval.num_row(), dtrain_train.num_col(), dtrain_eval.num_col())
num_class = (int)(np.max(dtrain_eval.get_label()) + 1)

num_class
param = {'max_depth':5, 'eta':0.5, 'num_class':num_class, 'objective':'multi:softmax'}

num_round = 10

bst = xgb.train(param, dtrain_train, num_round)

# bst = xgb.train(param, dtrain, num_round)



pred_label_eval = bst.predict(dtrain_eval)

true_label_eval = dtrain_eval.get_label()

print(np.sum(pred_label_eval == true_label_eval)/pred_label_eval.shape)

pred_label_train = bst.predict(dtrain_train)

true_label_train = dtrain_train.get_label()

print(np.sum(pred_label_train == true_label_train)/pred_label_train.shape)
pred_label_eval = bst.predict(dtrain_eval)

pred_label_eval
true_label_eval = dtrain_eval.get_label()

true_label_eval
np.sum(pred_label_eval == true_label_eval)/pred_label_eval.shape
pred_label_train = bst.predict(dtrain_train)

true_label_train = dtrain_train.get_label()

np.sum(pred_label_train == true_label_train)/pred_label_train.shape
x_train = df_train_data.values

y_train = df_train_label.values

x_train_train = x_train[:n_train, :]

x_train_eval = x_train[n_train:, :]

y_train_train = y_train[:n_train]

y_train_eval = y_train[n_train:]



x_test = df_test_data.values



print(x_test.shape, x_train_train.shape, x_train_eval.shape, y_train_train.shape, y_train_eval.shape)
x_train_train.shape
xgbclf = xgb.XGBClassifier(max_depth=3, learning_rate=0.1, 

                           objective='multi:softmax')



# xgbclf.fit(x_train_train, np.squeeze(y_train_train))

xgbclf.fit(x_train, np.squeeze(y_train))



pred_label_train_2 = xgbclf.predict(x_train_train)

print(np.sum(pred_label_train_2 == np.squeeze(y_train_train))/pred_label_train_2.size)



pred_label_eval_2 = xgbclf.predict(x_train_eval)

print(pred_label_eval_2)

print(np.squeeze(y_train_eval))

np.sum(pred_label_eval_2 == np.squeeze(y_train_eval))/pred_label_eval_2.size
xgbrf = xgb.XGBRFClassifier(max_depth=2, learning_rate=1, 

                           objective='multi:softmax')



xgbrf.fit(x_train_train, y_train_train)



pred_label_train_3 = xgbrf.predict(x_train_train)

print(np.sum(pred_label_train_3 == np.squeeze(y_train_train))/pred_label_train_3.size)



pred_label_eval_3 = xgbrf.predict(x_train_eval)

print(pred_label_eval_3)

print(np.squeeze(y_train_eval))

np.sum(pred_label_eval_3 == np.squeeze(y_train_eval))/pred_label_eval_3.size
from sklearn.model_selection import GridSearchCV



xgbclf_2 = xgb.XGBClassifier(objective='multi:softmax')

from sklearn.model_selection import GridSearchCV

gscv = GridSearchCV(xgbclf_2,

                   {'max_depth': [3,4,5,6],

                    'n_estimators': [90,100,110],

                     'learning_rate':[0.1,0.3,0.5,0.7,0.8,1]}, verbose=1, cv=2)



# gscv.fit(x_train_train, np.squeeze(y_train_train))

gscv.fit(x_train, np.squeeze(y_train))

print(gscv.best_score_)
pred_label_train_4 = gscv.predict(x_train_train)

print(np.sum(pred_label_train_4 == np.squeeze(y_train_train))/pred_label_train_4.size)



pred_label_eval_4 = gscv.predict(x_train_eval)

print(pred_label_eval_4)

print(np.squeeze(y_train_eval))

np.sum(pred_label_eval_4 == np.squeeze(y_train_eval))/pred_label_eval_4.size
from sklearn.ensemble import RandomForestClassifier



rfc = RandomForestClassifier(n_estimators=100)



# rfc.fit(x_train_train, np.squeeze(y_train_train))

rfc.fit(x_train, np.squeeze(y_train))



pred_label_train_5 = rfc.predict(x_train_train)

print(np.sum(pred_label_train_5 == np.squeeze(y_train_train))/pred_label_train_5.size)



pred_label_eval_5 = rfc.predict(x_train_eval)

print(pred_label_eval_5)

print(np.squeeze(y_train_eval))

np.sum(pred_label_eval_5 == np.squeeze(y_train_eval))/pred_label_eval_5.size
id_test = df_test.iloc[:, 0]

print(id_test.values.shape)
pred_label_test = rfc.predict(x_test)

# pred_label_test = np.array(bst.predict(dtest), dtype = np.int)



print(pred_label_test)

print(pred_label_test.shape)
out_test = np.stack([id_test, pred_label_test]).T

out_test.shape
print(out_test[:10])
df_out = pd.DataFrame(pred_label_test, index = id_test, columns = ['class'])

df_out.head()
df_out.to_csv('sub8.csv')