# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import warnings

warnings.filterwarnings("ignore")



import os

from os.path import join



import pandas as pd

import numpy as np



import missingno as msno



from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import mean_squared_error, make_scorer

from sklearn.model_selection import GridSearchCV   #Perforing grid search

from sklearn.preprocessing import LabelEncoder

import xgboost as xgb

import lightgbm as lgb



import matplotlib.pyplot as plt

import seaborn as sns



from shap import TreeExplainer, summary_plot
train_data_path = join('../input/2019-2nd-ml-month-with-kakr', 'train.csv')

sub_data_path = join('../input/2019-2nd-ml-month-with-kakr', 'test.csv')

data = pd.read_csv(train_data_path)

sub = pd.read_csv(sub_data_path)

print('train data dim : {}'.format(data.shape))

print('sub data dim : {}'.format(sub.shape))
# le = LabelEncoder()

# le.fit(data['zipcode'])

# le.fit(sub['zipcode'])



# data['zipcode'] = le.transform(data['zipcode'])

# sub['zipcode'] = le.transform(sub['zipcode'])
data = data.loc[data['id']!=456]

data = data.loc[data['id']!=2302]

data = data.loc[data['id']!=4123]

data = data.loc[data['id']!=7259]

data = data.loc[data['id']!=2777]



for df in [data,sub]:

#     df['date'] = df['date'].apply(lambda x: x[0:8])

#     df['date'] = df['date'].astype('int')

    df['date(new)'] = df['date'].apply(lambda x: int(x[4:8])+800 if x[:4] == '2015' else int(x[4:8])-400)

    df['how_old'] = df['date'].apply(lambda x: x[:4]).astype(int) - df[['yr_built', 'yr_renovated']].max(axis=1)

    df['date'] = df['date(new)']

    del df['date(new)']

    #del df['yr_renovated']

    df['yr_renovated'] = df['yr_renovated'].apply(lambda x: np.nan if x == 0 else x)

    df['yr_renovated'] = df['yr_renovated'].fillna(df['yr_built'])

    df['is_renovated'] = df['yr_renovated'] - df['yr_built']

    df['is_renovated'] = df['is_renovated'].apply(lambda x: 0 if x == 0 else 1)

    df['yr_built'] = df['yr_built'] - 1900

    df['sqft_floor'] = df['sqft_above'] / df['floors']

    df['floor_area_ratio'] = df['sqft_living'] / df['sqft_lot']

    df['rooms'] = df['bedrooms'] + df['bathrooms']

    

#     df['total_rooms'] = df['bedrooms'] + df['bathrooms']

#     df['sqft_ratio'] = df['sqft_living'] / df['sqft_lot']

    df['sqft_total_size'] = df['sqft_above'] + df['sqft_basement']

#     df['sqft_ratio15'] = df['sqft_living15'] / df['sqft_lot15'] 

    

    df['garret']=(df.floors%1==0.5).astype(int)

    df.loc[df.floors%1==0.5,'floors']=np.floor(df[df.floors%1==0.5].floors)

    df['exist_special']=df['garret']+df['waterfront']+df['is_renovated']

#     df['living_per_floors']=df['sqft_living']/df['floors']

#     df['total_score']=df['condition']+df['grade']+df['view']

#     df['diff_of_rooms']=np.abs(df['bedrooms']-df['bathrooms'])

#     df['diff_lots']=np.abs(df['sqft_lot15']-df['sqft_lot'])

#     df['diff_living']=np.abs(df['sqft_living15']-df['sqft_living'])

#     df['diff_living_per_floor']=(df.sqft_living15-df.sqft_living)/df.floors

    

#     del df['yr_built']

#     del df['sqft_lot15']

#     df['sqft_ratio_1'] = df['sqft_living'] / df['sqft_total_size'] 

#     df['sqft_ratio15'] = df['sqft_living15'] / df['sqft_lot15'] 





#data['per_price'] = data['price']/data['sqft_total_size']

data['per_price'] = data['price']/data['sqft_total_size']



zipcode_price = data.groupby(['zipcode'])['per_price'].agg({'mean','var'}).reset_index()

data = pd.merge(data,zipcode_price,how='left',on='zipcode')

sub = pd.merge(sub,zipcode_price,how='left',on='zipcode')



# for df in [data,sub]:

#     df['mean'] = df['mean'] * df['sqft_total_size']

#     df['var'] = df['var'] * df['sqft_total_size']



# skew_columns = ['bedrooms', 'sqft_living','sqft_lot','sqft_above','sqft_basement',

#                 'sqft_living15','sqft_lot15', 

#                 'mean', 'var']



skew_columns = ['bedrooms', 'sqft_lot', 'sqft_living', 'sqft_above', 'sqft_basement', 

                'sqft_living15', 'sqft_lot15', 'sqft_floor', 'floor_area_ratio',

                'sqft_total_size', 'mean', 'var', ]#'diff_lots', 'diff_living',]



for c in skew_columns:

    data[c] = np.log1p(data[c].values)

    sub[c] = np.log1p(sub[c].values)

    

data['price'] = np.log1p(data['price']) # price regularization
rows = (data.shape[1]+3) // 4

fig, axes = plt.subplots(rows, 4, figsize=(20, rows*5))

cols = data.columns



for r in range(rows):

    for c in range(4):

        index = 4 * r + c

        if index == len(cols):

            break

        sns.kdeplot(data[cols[index]], ax=axes[r, c])

        axes[r, c].set_title(cols[index], fontsize=20)
y = data['price']

del data['price']

del data['per_price']

train_len = len(data)

data = pd.concat((data, sub), axis=0)

sub_id = data['id'][train_len:]

del data['id']

sub = data.iloc[train_len:, :]

x = data.iloc[:train_len, :]
xgb_params = {

    'eta': 0.02,

    'max_depth': 6,

    'subsample': 0.9,

    'colsample_bytree': 0.6,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'silent': 1,

    'min_child_weight': 3, 

    'random_state': 0,

    'tree_method': 'gpu_hist',

}



dtrain = xgb.DMatrix(x, y)

dtest = xgb.DMatrix(sub)
def rmse_exp(predictions, dmat):

    labels = dmat.get_label()

    diffs = np.expm1(predictions) - np.expm1(labels)

    squared_diffs = np.square(diffs)

    avg = np.mean(squared_diffs)

    return ('rmse_exp', np.sqrt(avg))
cv_output = xgb.cv(xgb_params,

                   dtrain,                        

                   num_boost_round=5000,         # 학습 횟수

                   early_stopping_rounds=100,    # overfitting 방지

                   nfold=5,                      # 높을 수록 실제 검증값에 가까워지고 낮을 수록 빠름

                   verbose_eval=100,             # 몇 번째마다 메세지를 출력할 것인지

                   feval=rmse_exp,               # price 속성을 log scaling 했기 때문에, 다시 exponential

                   maximize=False,

                   show_stdv=False,              # 학습 동안 std(표준편차) 출력할지 말지

                   )



# scoring

best_rounds = cv_output.index.size

score = round(cv_output.iloc[-1]['test-rmse_exp-mean'], 2)



print(f'\nBest Rounds: {best_rounds}')

print(f'Best Score: {score}')





# plotting

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,5))

cv_output[['train-rmse-mean', 'test-rmse-mean']].plot(ax=ax1)

ax1.set_title('RMSE_log', fontsize=20)

cv_output[['train-rmse_exp-mean', 'test-rmse_exp-mean']].plot(ax=ax2)

ax2.set_title('RMSE', fontsize=20)



plt.show()
model = xgb.train(xgb_params, dtrain, num_boost_round=best_rounds)

y_pred = model.predict(dtest)

y_pred = np.expm1(y_pred)
fig, ax = plt.subplots(figsize=(10,10))

xgb.plot_importance(model, ax=ax)



plt.show()
submit = pd.read_csv("../input/sample_submission.csv")

submit['price'] = y_pred

submit.to_csv("sub_v3.csv",index=False)
def modelfit(model, x, y, cv_folds):

    xgb_param = model.get_xgb_params()

    dtrain = xgb.DMatrix(x, y)

    cvresult = xgb.cv(xgb_param, dtrain, num_boost_round=model.get_params()['n_estimators'], 

                      feval=rmse_exp,

                      verbose_eval=50, nfold=cv_folds, early_stopping_rounds=100)

    print(cvresult.shape[0])

#     model.set_params(n_estimators=cvresult.shape[0])

    

#     folds = KFold(n_splits=5, shuffle=True, random_state=511)

#     results = []

#     for n_fold, (trn_idx, val_idx) in enumerate(folds.split(x)):

#         trn_x, trn_y = x.ix[trn_idx], y[trn_idx] 

#         val_x, val_y = x.ix[val_idx], y[val_idx]

#         model.fit(trn_x.values, trn_y)

#         y_pred = model.predict(val_x.values)

#         rmse = np.sqrt(mean_squared_error(y_pred, val_y))

#         print(rmse)

#         results.append(rmse)

#     print("avg: {}".format(np.mean(results)))
xgb_final = xgb.XGBRegressor(

    max_depth=6, 

    learning_rate=0.02, 

    n_estimators=5000, 

    n_jobs=4, 

    gamma=0.0, 

    min_child_weight=3, 

    subsample=0.8, 

    colsample_bytree=0.8, 

    scale_pos_weight=1, 

    random_state=0,

)



modelfit(xgb_final, x, y, 5)
xgb_final = xgb.XGBRegressor(

    max_depth=6, 

    learning_rate=0.01, 

    n_estimators=3823, 

    n_jobs=4, 

    gamma=0.0, 

    min_child_weight=2, 

    subsample=0.8, 

    colsample_bytree=0.8, 

    scale_pos_weight=1, 

    random_state=0,

)



xgb_final.fit(x.values, y)
y_pred = xgb_final.predict(sub.values)

submit = pd.read_csv("../input/2019-2nd-ml-month-with-kakr/sample_submission.csv")

submit['price'] = np.expm1(y_pred)

submit.to_csv("sub_v3.csv",index=False)