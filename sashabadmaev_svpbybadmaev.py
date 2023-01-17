import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train  =pd.read_csv("/kaggle/input/santander-value-prediction-challenge/train.csv")

example=pd.read_csv("/kaggle/input/santander-value-prediction-challenge/sample_submission.csv")

test   =pd.read_csv("/kaggle/input/santander-value-prediction-challenge/test.csv")
#Очистка столбцов, содержащих константы

colsToRemove = []

for col in train.columns:

    if col != 'ID' and col != 'target':

        if train[col].std() == 0: 

            colsToRemove.append(col)

        

train.drop(colsToRemove, axis=1, inplace=True)



test.drop(colsToRemove, axis=1, inplace=True) 



print("Removed `{}` Constant Columns\n".format(len(colsToRemove)))

print(colsToRemove)
#Функция поиска повторяющихся столбцов

def duplicate_columns(frame):

    groups = frame.columns.to_series().groupby(frame.dtypes).groups

    dups = []



    for t, v in groups.items():



        cs = frame[v].columns

        vs = frame[v]

        lcs = len(cs)



        for i in range(lcs):

            ia = vs.iloc[:,i].values

            for j in range(i+1, lcs):

                ja = vs.iloc[:,j].values

                if np.array_equal(ia, ja):

                    dups.append(cs[i])

                    break



    return dups
colsToRemove = ['34ceb0081', '8d57e2749', '168b3e5bc', 'a765da8bc', 'acc5b709d']

print(colsToRemove)
train.drop(colsToRemove, axis=1, inplace=True) 



test.drop(colsToRemove, axis=1, inplace=True)



print("Removed `{}` Duplicate Columns\n".format(len(colsToRemove)))

print(colsToRemove)
#Функция очищения столбцов

def drop_sparse(train, test):

    flist = [x for x in train.columns if not x in ['ID','target']]

    for f in flist:

        if len(np.unique(train[f]))<2:

            train.drop(f, axis=1, inplace=True)

            test.drop(f, axis=1, inplace=True)

            #print(train[f])

    return train, test
print("Train set size: {}".format(train.shape))

print("Test set size: {}".format(test.shape))
%%time

train_df, test_df = drop_sparse(train, test)
X_train = train.drop(["ID", "target"], axis=1)

y_train = np.log1p(train["target"].values)



X_test = test.drop(["ID"], axis=1)



from sklearn.model_selection import train_test_split

dev_X, val_X, dev_y, val_y = train_test_split(X_train, y_train, test_size = 0.2, random_state = 42)



def run_xgb(train_X, train_y, val_X, val_y, test_X):

    params = {'objective': 'reg:linear', 

          'eval_metric': 'rmse',

          'eta': 0.001,

          'max_depth': 5, 

          'subsample': 0.6, 

          'colsample_bytree': 0.6,

          'alpha':0.001,

          'random_state': 42, 

          'silent': True}

    

    tr_data = xgb.DMatrix(train_X, train_y)

    va_data = xgb.DMatrix(val_X, val_y)

    

    watchlist = [(tr_data, 'train'), (va_data, 'valid')]

    

    model_xgb = xgb.train(params, tr_data, 2000, watchlist, maximize=False, early_stopping_rounds = 100, verbose_eval=100)

    

    dtest = xgb.DMatrix(test_X)

    xgb_pred_y = np.expm1(model_xgb.predict(dtest, ntree_limit=model_xgb.best_ntree_limit))

    

    return xgb_pred_y, model_xgb
import xgboost as xgb

pred_test_xgb, model_xgb = run_xgb(dev_X, dev_y, val_X, val_y, X_test)

print("XGB Training Completed...")
example['target'] = pred_test_xgb 

print(example.head())

example.to_csv('result.csv', index=False)