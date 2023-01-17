import warnings

warnings.filterwarnings("ignore")



import os

from os.path import join



import pandas as pd

import numpy as np



import missingno as msno



from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import KFold, cross_val_score

import xgboost as xgb

import lightgbm as lgb



import matplotlib.pyplot as plt

import seaborn as sns
train_data_path = join('../input', 'train.csv')

sub_data_path = join('../input', 'test.csv')
data = pd.read_csv(train_data_path)

sub = pd.read_csv(sub_data_path)

print('train data dim : {}'.format(data.shape))

print('sub data dim : {}'.format(sub.shape))
y = data['price']



del data['price']
train_len = len(data)

data = pd.concat((data, sub), axis=0)
data.head()
msno.matrix(data)
for c in data.columns:

    print('{} : {}'.format(c, len(data.loc[pd.isnull(data[c]), c].values)))
sub_id = data['id'][train_len:]

del data['id']

data['date'] = data['date'].apply(lambda x : str(x[:6])).astype(str)
fig, ax = plt.subplots(10, 2, figsize=(20, 60))



# id 변수는 제외하고 분포를 확인합니다.

count = 0

columns = data.columns

for row in range(10):

    for col in range(2):

        sns.kdeplot(data[columns[count]], ax=ax[row][col])

        ax[row][col].set_title(columns[count], fontsize=15)

        count+=1

        if count == 19 :

            break
skew_columns = ['bedrooms', 'sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement']



for c in skew_columns:

    data[c] = np.log1p(data[c].values)
fig, ax = plt.subplots(3, 2, figsize=(10, 15))



count = 0

for row in range(3):

    for col in range(2):

        if count == 5:

            break

        sns.kdeplot(data[skew_columns[count]], ax=ax[row][col])

        ax[row][col].set_title(skew_columns[count], fontsize=15)

        count+=1



sub = data.iloc[train_len:, :]

x = data.iloc[:train_len, :]
gboost = GradientBoostingRegressor(random_state=2019)

xgboost = xgb.XGBRegressor(random_state=2019)

lightgbm = lgb.LGBMRegressor(random_state=2019)



models = [{'model':gboost, 'name':'GradientBoosting'}, {'model':xgboost, 'name':'XGBoost'},

          {'model':lightgbm, 'name':'LightGBM'}]
def get_cv_score(models):

    kfold = KFold(n_splits=5, random_state=2019).get_n_splits(x.values)

    for m in models:

        print("Model {} CV score : {:.4f}".format(m['name'], np.mean(cross_val_score(m['model'], x.values, y)), 

                                             kf=kfold))
get_cv_score(models)
def AveragingBlending(models, x, y, sub_x):

    for m in models : 

        m['model'].fit(x.values, y)

    

    predictions = np.column_stack([

        m['model'].predict(sub_x.values) for m in models

    ])

    return np.mean(predictions, axis=1)
y_pred = AveragingBlending(models, x, y, sub)
sub = pd.DataFrame(data={'id':sub_id,'price':y_pred})
sub.to_csv('submission.csv', index=False)