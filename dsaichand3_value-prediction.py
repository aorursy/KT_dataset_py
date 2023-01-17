import numpy as np 

import pandas as pd



from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor

from catboost import CatBoostRegressor



%matplotlib inline
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
dataset = pd.read_csv("../input/santander-value-prediction-challenge/train.csv")

dataset.shape
dataset.head(5)
dataset.describe()
dataset.isnull().sum()
dataset.dtypes
dataset.dtypes.value_counts()
columns = dataset.std() == 0

const_columns = columns.iloc[[i for i, x in enumerate(columns) if x]]

dataset.drop(const_columns.index, axis = 1, inplace = True)
def drop_sparse(train):

    sparse_columns = []

    flist = dataset.columns[2:]

    for f in flist:

        if len(np.unique(train[f]))<2:

            sparse_columns.append(f)

    return sparse_columns



sparse_columns = drop_sparse(dataset)

dataset.drop(sparse_columns, axis = 1, inplace = True)
def duplicate_columns(dataset):

    groups = dataset.columns.to_series().groupby(dataset.dtypes).groups

    my_dict = {}

    duplicate_features = []

    

    for d_type, columns in groups.items():

        columns_group = dataset[columns]

        list_of_column_names = dataset[columns].columns

        length = len(columns)

        

        for i in range(length):

            a = tuple(columns_group.iloc[:, i])

            if a in my_dict:

                duplicate_features.append(list_of_column_names[i])

            else:

                my_dict[a] = list_of_column_names[i]

            

    return duplicate_features



duplicate_features = duplicate_columns(dataset)

dataset.drop(sparse_columns, axis = 1, inplace = True)
dataset = dataset.drop(['ID'], axis = 1)

X = dataset.iloc[:,1:]       

Y = dataset.iloc[:,0]  
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.21, random_state=42)
xb_model = XGBRegressor(colsample_bytree=0.055, colsample_bylevel =0.5, 

                             gamma=1.5, learning_rate=0.01, max_depth=32, 

                             objective='reg:linear',booster='gbtree',

                             min_child_weight=57, n_estimators=800, reg_alpha=0, 

                             reg_lambda = 0,eval_metric = 'rmse', subsample=0.7, 

                             silent=1, n_jobs = -1, early_stopping_rounds = 14,

                             random_state =7, nthread = -1)

xb_model.fit(x_train, y_train,

             eval_set=[(x_test, y_test)], verbose=False)
# Predicting the values form the test_set

y_pred = xb_model.predict(x_test)



# the root-mean squared error for predictions

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

rmse
# Comparing the predictions rmse with benchmark rmse

y_mean = [y_test.mean()] * y_test.shape[0]



rmse_benchmarch = np.sqrt(mean_squared_error(y_mean, y_pred))

print(rmse_benchmarch)
# Loading the test dataset for predicting the target feature

test_dataset = pd.read_csv("../input/santander-value-prediction-challenge/test.csv")

test_id_s = test_dataset['ID']

test_dataset = test_dataset[dataset.columns[1:]]

test_pred_xgb = xb_model.predict(test_dataset)

test_pred_xgb = np.clip(test_pred_xgb, 1, float('inf'))
cb_model = CatBoostRegressor(iterations=1000,

                             learning_rate=0.05,

                             depth=10,

                             eval_metric='RMSE',

                             random_seed = 42,

                             bagging_temperature = 0.2,

                             od_type='Iter',

                             metric_period = 50,

                             od_wait=20)
cb_model.fit(x_train, y_train,

             eval_set=(x_test, y_test),

             use_best_model=True,

             verbose=50)
test_pred_cb = cb_model.predict(test_dataset)

test_pred_cb = np.clip(test_pred_cb, 0, float('inf'))
final_preds = (test_pred_xgb * 0.5 + test_pred_cb * 0.3)

pred_df = pd.DataFrame({'ID':test_id_s, 'target': final_preds})

pred_df.to_csv("submission.csv", float_format="%.10g", index = False)