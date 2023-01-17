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
df=pd.read_csv("/kaggle/input/santander-value-prediction-challenge/train.csv")
db=pd.read_csv("/kaggle/input/santander-value-prediction-challenge/sample_submission.csv")
dc=pd.read_csv("/kaggle/input/santander-value-prediction-challenge/test.csv")
#Использованный код https://www.kaggle.com/samratp/lightgbm-xgboost-catboost
#Здесь идет очистка столбцов, содержащих константы (типо каждая строка в столбце = 0)
colsToRemove = []
for col in df.columns:
    if col != 'ID' and col != 'target':
        if df[col].std() == 0: 
            colsToRemove.append(col)
        
# remove constant columns in the training set
df.drop(colsToRemove, axis=1, inplace=True)

# remove constant columns in the test set
dc.drop(colsToRemove, axis=1, inplace=True) 

print("Removed `{}` Constant Columns\n".format(len(colsToRemove)))
print(colsToRemove)
#Функция поиска... повторяющихся столбцов? Долго работает, поэтому я просто скопирую (и удалю) названия удаленных дубликатов (которые автор уже нашел)
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
# remove duplicate columns in the training set
df.drop(colsToRemove, axis=1, inplace=True) 

# remove duplicate columns in the testing set
dc.drop(colsToRemove, axis=1, inplace=True)

print("Removed `{}` Duplicate Columns\n".format(len(colsToRemove)))
print(colsToRemove)
#Функция очищения столбцов которые видимо... 
def drop_sparse(train, test):
    flist = [x for x in train.columns if not x in ['ID','target']]
    for f in flist:
        if len(np.unique(train[f]))<2:
            train.drop(f, axis=1, inplace=True)
            test.drop(f, axis=1, inplace=True)
            #print(train[f])
    return train, test
%%time
train_df, test_df = drop_sparse(df, dc)
print("Train set size: {}".format(df.shape))
print("Test set size: {}".format(dc.shape))
X_train = df.drop(["ID", "target"], axis=1)
y_train = np.log1p(df["target"].values)

X_test = dc.drop(["ID"], axis=1)
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
db['target'] = pred_test_xgb 
print(db.head())
db.to_csv('sub_lgb_xgb_cat.csv', index=False)
