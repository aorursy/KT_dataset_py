import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas import DataFrame

import matplotlib.pyplot as plt

from torch.autograd import Variable
import os



print(os.listdir("../input"))
raw_data = pd.read_csv('../input/train.csv')
raw_data.describe()
raw_data.head(10)
numeric_colmuns = []

numeric_colmuns.extend(list(raw_data.dtypes[raw_data.dtypes == np.int64].index))

numeric_colmuns.extend(list(raw_data.dtypes[raw_data.dtypes == np.float64].index))
numeric_colmuns

for i in numeric_colmuns:

    if 'index' in str(i):

        numeric_colmuns.remove(i)
numeric_colmuns.remove('total_price')

numeric_colmuns.append('total_price')
numeric_data = DataFrame(raw_data, columns=numeric_colmuns)
numeric_data.describe()
numeric_data.head(10)
nan_columns = np.any(pd.isna(numeric_data), axis = 0)

nan_columns = list(nan_columns[nan_columns == True].index)
nan_columns
for i in nan_columns:

    numeric_data[i] = numeric_data[i].fillna(0)
nan_columns = np.any(pd.isna(numeric_data), axis = 0)

nan_columns = list(nan_columns[nan_columns == True].index)
nan_columns
import torch

import torch.nn as nn
numeric_x_columns = list(numeric_data.columns)

numeric_x_columns.remove('total_price')

'''

numeric_x_columns.remove('building_material')

numeric_x_columns.remove('building_use')

numeric_x_columns.remove('parking_way')

'''

numeric_y_columns = ['total_price']
numeric_x_df = DataFrame(numeric_data, columns=numeric_x_columns)

numeric_y_df = DataFrame(numeric_data, columns=numeric_y_columns)
numeric_x = torch.tensor(numeric_x_df.values, dtype=torch.float)

numeric_y = torch.tensor(numeric_y_df.values, dtype=torch.float)
means, maxs, mins = dict(), dict(), dict()
for col in numeric_data:

    means[col] = numeric_data[col].mean()

    maxs[col] = numeric_data[col].max()

    mins[col] = numeric_data[col].min()
numeric_data2 = (numeric_data - numeric_data.mean()) / (numeric_data.max() - numeric_data.min())
nan_columns = np.any(pd.isna(numeric_data), axis = 0)

nan_columns = list(nan_columns[nan_columns == True].index)

print(nan_columns)

for i in nan_columns:

    numeric_data2[i] = numeric_data2[i].fillna(0)

    
numeric_x_df = DataFrame(numeric_data2, columns=numeric_x_columns)

numeric_y_df = DataFrame(numeric_data2, columns=numeric_y_columns)

numeric_x_df.describe()

#numeric_x_df.head(10)

#numeric_y_df.describe()
import xgboost as xgb

dtrain = xgb.DMatrix(numeric_x_df, label = numeric_y_df)



params = {"max_depth":6, "eta":0.03}

model = xgb.cv(params, dtrain,  num_boost_round=2000, early_stopping_rounds=500)
model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()

model_xgb = xgb.XGBRegressor(n_estimators=1000, max_depth=6, learning_rate=0.03) #the params were tuned using xgb.cv

model_xgb.fit(numeric_x_df, numeric_y_df)
raw_test_data = pd.read_csv('../input/test.csv')
raw_test_data.describe()
raw_test_data.head(10)
test_x = DataFrame(raw_test_data,columns=numeric_x_columns)

t_id =  DataFrame(raw_test_data)
for col in numeric_x_columns:

    test_x[col].fillna(0)

#test_x=test_x.drop('total_price',axis=1)

test_x.describe()
test_x.describe()
for col in test_x.columns:

    test_x[col] = (test_x[col] - means[col]) / (maxs[col] - mins[col])

test_x.describe()
xgb_preds = np.expm1(model_xgb.predict(test_x))

predictions = pd.DataFrame({"xgb":xgb_preds})

solution = pd.DataFrame({"building_id":t_id.building_id, "total_price":xgb_preds* (maxs['total_price'] - mins['total_price']) + means['total_price']})

print(solution)

solution.to_csv("./ridge_sol.csv", index = False)