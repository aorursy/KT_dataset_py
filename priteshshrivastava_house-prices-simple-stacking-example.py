import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestRegressor

from sklearn import linear_model

from sklearn.model_selection import train_test_split

import os

import math



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_df = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

test_df = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

train_df.head()
train_df['SalePrice'] = np.log(train_df['SalePrice'])
import fastai_structured as fs

fs.train_cats(train_df)

fs.apply_cats(test_df, train_df)
nas = {}

df_trn, y_trn, nas = fs.proc_df(train_df, 'SalePrice', na_dict=nas)   ## Avoid creating NA columns as total cols may not match later

df_test, _, _ = fs.proc_df(test_df, na_dict=nas)

df_trn.head()
def rmse(x,y): return math.sqrt(((x-y)**2).mean())



def print_score(m):

    res = [rmse(m.predict(train_X), train_y), rmse(m.predict(val_X), val_y),     ## RMSE of log of prices

                m.score(train_X, train_y), m.score(val_X, val_y)]

    #if hasattr(m, 'oob_score_'): res.append(m.oob_score_)

    print(res)
train_X, val_X, train_y, val_y = train_test_split(df_trn, y_trn, test_size=0.5, random_state=42)
model1 = linear_model.LinearRegression()

model2 = RandomForestRegressor()
model1.fit(train_X, train_y)

model2.fit(train_X, train_y)
print_score(model1)
print_score(model2)
preds1 = model1.predict(val_X)

preds2 = model2.predict(val_X)
test_preds1 = model1.predict(df_test)

test_preds2 = model2.predict(df_test)
stacked_predictions = np.column_stack((preds1, preds2))

stacked_test_predictions = np.column_stack((test_preds1, test_preds2))
meta_model = linear_model.LinearRegression()
meta_model.fit(stacked_predictions, val_y)
final_predictions = meta_model.predict(stacked_test_predictions)
submission = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')

submission.head()
submission['SalePrice'] = np.exp(final_predictions)   ## Convert log back 

submission.to_csv('stacking_example.csv', index=False)