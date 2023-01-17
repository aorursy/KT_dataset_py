import warnings

warnings.filterwarnings("ignore")



import os

from os.path import join



import pandas as pd

import numpy as np

import math



# import missingno as msno



from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import KFold, cross_val_score

from sklearn import show_versions



import xgboost as xgb

import lightgbm as lgb



import matplotlib.pyplot as plt

import seaborn as sns



show_versions()
train_data_path = join('../input', 'train.csv')

sub_data_path = join('../input', 'test.csv')



train_data = pd.read_csv(train_data_path)

data = train_data

sub = pd.read_csv(sub_data_path)

print('train data dim : {}'.format(data.shape))

print('sub data dim : {}'.format(sub.shape))
train_y = data['price']

del data['price']



train_len = len(data)

data = pd.concat((data, sub), axis=0).reset_index(drop=True)

print('train_len: {}, data_len: {}'.format(train_len, len(data)))

# data['renovate'] = data['yr_renovated'] - data['yr_built']

data['renovate'] = data.apply(lambda x: x['yr_renovated'] - x['yr_built'] if x['yr_renovated'] > 0 else 0, axis=1)



dummied_column = ["waterfront", "view", "condition"]

data = pd.get_dummies(data, columns=dummied_column)



# data['bath_bed'] = data.apply(lambda x: x['bathrooms'] / x['bedrooms'] if x['bedrooms'] > 0 else 0, axis=1)

# data['bath_floors'] = data['bathrooms'] / data['floors']

# data['bed_floors'] = data['bedrooms'] / data['floors']



data['sqft_bath'] = data['sqft_living'] / data['bathrooms']

data['sqft_bed'] = data['sqft_living'] / data['bedrooms']



data['sqft15_bath'] = data['sqft_living15'] / data['bathrooms']

data['sqft15_bed'] = data['sqft_living15'] / data['bedrooms']



data = data.replace([np.inf, -np.inf], np.nan)

data = data.fillna(0.0)

    

#data.head()
sub_id = data['id'][train_len:]

del data['id']

data['date'] = data['date'].apply(lambda x : str(x[:6])).astype(str)
# rows = math.ceil(len(data.columns)/2)

# fig, ax = plt.subplots(rows, 2, figsize=(20, 60))



# # id 변수는 제외하고 분포를 확인합니다.



# # categorical 데이터는 분리해서 보는 것이 좋겠다.

# count = 0

# columns = data.columns

# for row in range(rows):

#     for col in range(2):

#         sns.kdeplot(data[columns[count]], ax=ax[row][col])

#         ax[row][col].set_title(columns[count], fontsize=15)

#         count+=1

#         if count == len(data.columns) :

#             break
# skew_columns = ['bedrooms', 'sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement']

# # skew_columns = ['sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement']

# for c in skew_columns:

#     # data[c] = np.log1p(data[c].values)

#     data[c] = np.log1p(data[c].values)
# rows = math.ceil(len(skew_columns)/2)

# fig, ax = plt.subplots(rows, 2, figsize=(10, rows*5))



# count = 0

# for row in range(3):

#     for col in range(2):

#         if count == len(skew_columns):

#             break

#         sns.kdeplot(data[skew_columns[count]], ax=ax[row][col])

#         ax[row][col].set_title(skew_columns[count], fontsize=15)

#         count+=1
# data['sqft_living_ratio'] = data['sqft_living'] / (data['sqft_living'] + data['sqft_lot'])

# data['sqft_lot_ratio'] = data['sqft_lot'] / (data['sqft_living'] + data['sqft_lot'])

# data['sqft_living15_ratio'] = data['sqft_living15'] / (data['sqft_living15'] + data['sqft_lot15'])

# data['sqft_lot15_ratio'] = data['sqft_lot15'] / (data['sqft_living15'] + data['sqft_lot15'])
lat_min = data['lat'].min() if data['lat'].min() < sub['lat'].min() else sub['lat'].min()

lat_max = data['lat'].max() if data['lat'].max() > sub['lat'].max() else sub['lat'].max()



long_min = data['long'].min() if data['long'].min() < sub['long'].min() else sub['long'].min()

long_max = data['long'].max() if data['long'].max() > sub['long'].max() else sub['long'].max()



print("lat range is {} ~ {}, distance:{}".format(lat_min, lat_max, lat_max - lat_min))

print("long range is {} ~ {}, distance:{}".format(long_min, long_max, long_max - long_min))



#data['latlong'] = data.apply(lambda x: round((x['lat']-lat_min)*100)*100 + round((x['long']-long_min)*100), axis=1)

data['longlat'] = data.apply(lambda x: round((x['long']-long_min)*100)*100 + round((x['lat']-lat_min)*100), axis=1)

def feature_weight(train_data, key_column, grouped_column) :

    grouped_mean = train_data.groupby([key_column])[grouped_column].mean().reset_index()

    grouped_mean_column = 'grouped_mean'

    grouped_mean.columns = [key_column, grouped_mean_column] 

    

    # [grouped_column, grouped_mean]

    tmp = pd.merge(train_data[[key_column, grouped_column]], grouped_mean, on=key_column, how='inner')

    weight_column_name = grouped_column + '_w'

    tmp[weight_column_name] = tmp[grouped_column] / tmp[grouped_mean_column]

    tmp = tmp.fillna(1.0)

    return (tmp, weight_column_name)
# for f in ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 

#           'grade', 'sqft_above', 

#           'sqft_basement', 'yr_renovated', 'renovate',

#           'yr_built', 'sqft_living15', 'sqft_lot15'] :

#     dd, weight_column = feature_weight(data, 'zipcode', f)

#     data = pd.concat([data, dd[weight_column]], axis=1)
is_null = False

for c in data.columns:

    null_count = len(data.loc[pd.isnull(data[c]), c].values)

    na_count = len(data.loc[pd.isna(data[c]), c].values)

    #nan_count = len(data.loc[pd.isnan(data[c]), c].values) 

    #inf_count = len(data.loc[pd.isinf(data[c]), c].values)     

    if null_count + na_count > 0 : # or nan_count > 0 or inf_count > 0:

        print('{} : null({}), na({})'.format(c, null_count, na_count))

        is_null = True

if is_null == False :

    print('No null value in all columns')

    

 
data[data.isin([np.nan, np.inf, -np.inf]).any(1)]
sub_x = data.iloc[train_len:, :]

train_x = data.iloc[:train_len, :] 
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error



gboost = GradientBoostingRegressor(random_state =2019)

xgboost = xgb.XGBRegressor(random_state =2019)

lightgbm = lgb.LGBMRegressor(objective='regression', random_state =2019)



models = [{'model':gboost, 'name':'GradientBoosting'}

          , {'model':xgboost, 'name':'XGBoost'}

          , {'model':lightgbm, 'name':'LightGBM'}

         ]



max_features = [16]

param_grid_narrow = [

    {"max_depth":[6, 7]

     , "n_estimators":[10, 20]

     , "max_features":max_features

    }

    #, {"bootstrap": [False], "max_depth":[4, 5, 6, 7, 8], "n_estimators": [3, 10], "max_features":[4, 8, 16, 32]}, 

]



if train_x.shape[1] > 16 :

    if train_x.shape[1] < 32 :

        max_features.append(train_x.shape[1])

    elif train_x.shape[1] >= 32 :

        if train_x.shape[1] - 32 > 16:

            max_features.append(32)    

        max_features.append(train_x.shape[1])

    

param_grid_wide = [

    {"max_depth": [5, 6]

     , "n_estimators": [800, 900, 1000, 1100, 1200]

     , "max_features": max_features

    }

    #, {"bootstrap": [False], "max_depth":[4, 5, 6, 7, 8], "n_estimators": [3, 10], "max_features":[4, 8, 16, 32]}, 

]



def get_best_param(model, param_grid, train_x, train_y) :

    _search = GridSearchCV(model['model']

                            , param_grid

                            , cv = 5

                            , scoring = "neg_mean_squared_error"

                            , return_train_score = True

                            , verbose = 4

                            , n_jobs = -1

                        )



    _search.fit(train_x.values, train_y)

    preds = _search.predict(train_x.values)

    rmse = np.sqrt(mean_squared_error(train_y, preds))

    print("{}' rmse: {:.5f}, cv score: {:.5f}".format(model['name'], rmse, np.sqrt(-_search.best_score_)))

    

    return _search



print("### data columns")

print(train_x.columns)



param_grid = param_grid_wide

print("### param_grid")

print(param_grid)



for model in models:

    # _best_model = get_best_param(model, param_grid_narrow, train_x, train_y)    

    _best_model = get_best_param(model, param_grid, train_x, train_y)

    print("{}'s best param: {}".format(model['name'], _best_model.best_params_))

    model['fit_model'] = _best_model
pd.set_option('display.max_columns', 500)

train_x.head()
model_feature_importance = {}

for m in models : 

    model_feature_importance[m['name']] = dict(zip(data.columns, m['fit_model'].best_estimator_.feature_importances_))



model_feature_importance_df = pd.DataFrame(model_feature_importance)

model_feature_importance_df
fig, ax = plt.subplots(1, 3, figsize=(20, 20))

count = 0

model_feature_importance_df.reset_index()



for m in models : 

    model = m['name']

    d = model_feature_importance_df.sort_values(by=model, ascending=False)

    sns.barplot(x = model

                , y = d.index

                , data = d

                , ax = ax[count])

    ax[count].set_title(model, fontsize=15)

    count += 1
from scipy import stats



def AveragingBlending(models, x, y, sub_x):

    rmse = []

    for m in models : 

        preds = m['fit_model'].predict(x.values)

        model_rmse = np.sqrt(mean_squared_error(y, preds))

        print("{}' rmse: {:.5f}".format(m['name'], model_rmse))

        rmse.append(model_rmse)

    

    predictions = np.column_stack([

        m['fit_model'].predict(sub_x.values) for m in models

    ])

    

    return np.mean(predictions, axis=1)
from scipy import stats



def AveragingBlending2(models, x, y, sub_x):

    rmse = []

    for m in models : 

        preds = m['fit_model'].predict(x.values)

        model_rmse = np.sqrt(mean_squared_error(y, preds))

        print("{}' rmse: {:.5f}".format(m['name'], model_rmse))

        rmse.append(model_rmse)

    

    rmse_sum = np.sum(np.square(rmse))

    weight = (rmse_sum - np.square(rmse)) / rmse_sum

    print("weight: ", weight)

    

    predictions = np.column_stack([

        m['fit_model'].predict(train_x.values) for m in models

    ])

    

    mean_prediction = np.mean(predictions, axis=1)

    weighted_mean_prediction = np.average(predictions, weights=weight, axis=1)

    rmse_mean_prediction = np.sqrt(mean_squared_error(y, mean_prediction))

    rmse_weighted_mean_prediction = np.sqrt(mean_squared_error(y, weighted_mean_prediction))

    print("mean prediction rmse: {:.5f}".format(rmse_mean_prediction))

    print("weighted mean prediction rmse: {:.5f}".format(rmse_weighted_mean_prediction))



    predictions = np.column_stack([

        m['fit_model'].predict(sub_x.values) for m in models

    ])



    if rmse_mean_prediction < rmse_weighted_mean_prediction :

        return np.mean(predictions, axis=1)

    else :

        return np.average(predictions, weights=weight, axis=1)

    

AveragingBlending2(models, train_x, train_y, sub_x)
y_pred = AveragingBlending(models, train_x, train_y, sub_x)

sub = pd.DataFrame(data={'id':sub_id,'price':y_pred})

sub.to_csv('submission.csv', index=False)