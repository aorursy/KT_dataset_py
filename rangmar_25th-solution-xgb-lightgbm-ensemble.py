# 분석 기본 도구

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import gc

from sklearn.preprocessing import LabelEncoder

from sklearn.decomposition import PCA

from sklearn.cluster import KMeans

import xgboost as xgb

import lightgbm as lgb

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import GridSearchCV

from lightgbm import LGBMRegressor

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import KFold

%matplotlib inline



def haversine_array(lat2, lng2):

    lat1, lng1 = 47.63, -122.22

    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))

    AVG_EARTH_RADIUS = 6371  # in km

    lat = lat2 - lat1

    lng = lng2 - lng1

    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2

    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))

    return h



def rmse_exp(predictions, dmat):

    labels = dmat.get_label()

    diffs = np.expm1(predictions) - np.expm1(labels)

    squared_diffs = np.square(diffs)

    avg = np.mean(squared_diffs)

    return ('rmse_exp', np.sqrt(avg))



def print_best_params(model, params):

    grid_model = GridSearchCV(

        model, 

        param_grid = params,

        scoring='neg_mean_squared_error',

        cv=5,

        n_jobs=-1

    )



    grid_model.fit(X_train, y_train)

    rmse = np.sqrt(-1*grid_model.best_score_)

    print(

        '{0} 5 CV 시 최적 평균 RMSE 값 {1}, 최적 alpha:{2}'.format(model.__class__.__name__, np.round(rmse, 6), grid_model.best_params_))

    return grid_model.best_estimator_



def zipcode_groupby(train, test, group_col, colname, agg_method) :

    new_colname = 'price_per'+'_'+colname

    #new_colname2 = colname+'mean'

    

    train[new_colname] = train['price']/train[colname]

    price_per_temp = train.groupby([group_col])[new_colname].agg(agg_method)

    price_per_temp.columns = ['{}_{}'.format(new_colname, m) for m in agg_method]

    price_per_temp = price_per_temp.reset_index()

    #price_per_temp.rename(columns={'mean':new_colname2}, inplace=True)

    train = pd.merge(train, price_per_temp, how='left', on=group_col)

    test = pd.merge(test, price_per_temp, how='left', on=group_col)

    

    del train[new_colname]

    

    return train, test



def groupby_helper(df, group_col, target_col, agg_method, prefix_param=None):

    try:

        prefix = get_prefix(group_col, target_col, prefix_param)

        print(group_col, target_col, agg_method)

        group_df = df.groupby(group_col)[target_col].agg(agg_method)

        group_df.columns = ['{}_{}'.format(prefix, m) for m in agg_method]

    except BaseException as e:

        print(e)

    return group_df.reset_index()



def get_prefix(group_col, target_col, prefix=None):

    if isinstance(group_col, list) is True:

        g = '_'.join(group_col)

    else:

        g = group_col

    if isinstance(target_col, list) is True:

        t = '_'.join(target_col)

    else:

        t = target_col

    if prefix is not None:

        return prefix + '_' + g + '_' + t

    return g + '_' + t



def category_feature_distribution(train, col, target='price'):

    fig, ax = plt.subplots(1, 2, figsize=(16,4))

    

    for c in sorted(train[col].unique()):

        sns.distplot(train.loc[train[col]==c, target], ax=ax[0])

    ax[0].legend(sorted(train[col].unique()))

    ax[0].set_title(f'{col} {target} distribution')



    sns.boxplot(x=col, y=target, data=train, ax=ax[1])

    ax[1].set_title(f'{col} vs {target}')

    

    plt.show()

    

def haversine_array_new(lat1_raw, lng1_raw, lat2, lng2):

    lat1, lng1 = lat1_raw, lng1_raw

    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))

    AVG_EARTH_RADIUS = 6371  # in km

    lat = lat2 - lat1

    lng = lng2 - lng1

    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2

    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))

    return h



def haversine_array_low(lat2, lng2):

    lat1, lng1 = 47.382, -122.247

    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))

    AVG_EARTH_RADIUS = 6371  # in km

    lat = lat2 - lat1

    lng = lng2 - lng1

    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2

    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))

    return h
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

data = pd.merge(train, test, how='outer')



train_raw = train

train_len = len(train)



data['min_haver'] = np.nan

for i in range(len(data)) :

    temp_lat, temp_lng = data[['lat', 'long']].loc[i,'lat'], data[['lat', 'long']].loc[i,'long']

    temp_all_df = data.drop([i], 0)

    temp_coord_df = haversine_array_new(temp_lat, temp_lng, temp_all_df['lat'], temp_all_df['long'])

    temp_min = temp_coord_df.min()

    data.loc[i,'min_haver'] = temp_min    
######## zipcode 라벨링

train = data.iloc[:train_len,:]

test = data.iloc[train_len:,:]



for df in [train, test] :

    #df['zip_1'] = df['zipcode'].apply(lambda x : str(x)[2]).astype(int)

    df['zip_12'] = df['zipcode'].apply(lambda x : str(x)[2:4]).astype(int)

    #df['zip_2'] = df['zipcode'].apply(lambda x : str(x)[3]).astype(int)

    #df['zip_23'] = df['zipcode'].apply(lambda x : str(x)[3:5]).astype(int)

    #df['zip_3'] = df['zipcode'].apply(lambda x : str(x)[4]).astype(int)



le = LabelEncoder()



le.fit(train['zipcode'])

le.fit(test['zipcode'])



train['zipcode'] = le.transform(train['zipcode'])

test['zipcode'] = le.transform(test['zipcode'])



train['sqft_total_size'] = train['sqft_above'] + train['sqft_basement'] # 총 주거 면적    

test['sqft_total_size'] = test['sqft_above'] + test['sqft_basement']



####### zipcode_groupby 단가 변수 생성

train['price_per_land_area'] = train['price'] / (train['sqft_living'])

price_per_ft = train.groupby(['zipcode'])['price_per_land_area'].agg({'mean', 'std', 'count'}).reset_index()

train = pd.merge(train, price_per_ft, how='left', on='zipcode')

test = pd.merge(test, price_per_ft, how='left', on='zipcode')

del train['price_per_land_area']



####### train, test set 지정



X_train = train.drop(['id', 'price'], 1)

y_train = train['price']

y_train = np.log1p(y_train)

X_test = test.drop(['id', 'price'], axis=1)



####### KMeans Clustering

km_n = 120

km = KMeans(n_clusters=km_n, random_state=2019)

km.fit(X_train[['lat', 'long']])#



######## 이외 Feature Engineering

for df in [X_train, X_test]:

    df['date(new)'] = df['date'].apply(lambda x: int(x[4:8])+800 if x[:4] == '2015' else int(x[4:8])-400) # 날짜 줄 세우기

    df['how_old'] = df['date'].apply(lambda x: x[:4]).astype(int) - df[['yr_built', 'yr_renovated']].max(axis=1) # 얼마나 됐는지 연식

    df['yr_built'] = df['yr_built'] - 1900 # 건축년도 1900년도로부터 얼마나 됐는지 

    

    # sqft 관련

    df['sqft_diff'] = df['sqft_living15'] - df['sqft_living']

    df['sqft_living_lot_diff'] = df['sqft_lot'] - df['sqft_living']

    #df['sqft_living_lot_div'] = df['sqft_living'] / df['sqft_lot']

    del df['sqft_lot15'], df['yr_renovated'], df['sqft_lot'],df['date']#, df['waterfront']#, df['view'], df['condition']

    

    #df['sqft_living_all'] = df['sqft_living'] + df['sqft_living15']

    #df['living_div'] = df['sqft_living15'] / df['sqft_living_all']    

    

    # 방 관련

    df['sqft_bedrooms'] = df['sqft_total_size'] / (df['bedrooms'] + 1) # 방하나당 면적 : bedrooms가 0인 자료 꽤 있음 

    

    

    # KMeans 클러스터링

    km_col_name = 'km'+ '_' + str(km_n)

    df[km_col_name] = km.predict(df[['lat', 'long']])

    

    # 레벨 관련

    df['sum_level'] = df['grade'] + df['view'] + df['condition'] # 등급 총합

    df['multi_level'] = df['grade'] * (df['view']+1) * df['condition']

    

    df['condition_2'] = df['condition'].apply(lambda x : 0 if x < 3 else x)

    df['sum_level_2'] = df['view'] + df['condition_2'] + df['grade']

    

    del df['condition_2']

    

    df['low_cond']=df['condition'].apply(lambda x : 0 if x <= 2 else 1)

    df['low_view'] = df['view'].apply(lambda x : 0 if x == 0 else 1)

    df['low_bath'] = df['bathrooms'].apply(lambda x: 0 if x < 1 else 1)

    df['low_bed'] = df['bedrooms'].apply(lambda x : 0 if x ==1 else 1)

    df['low_grade'] = df['grade'].apply(lambda x : 0 if x <= 6 else 1)

    df['low_all'] = (df['low_cond'] + df['low_view']+df['low_bath'] + df['low_bed'] + df['low_grade']) + df['waterfront']

    

    del df['low_cond'], df['low_view'], df['low_bath'], df['low_bed'], df['low_grade']

   

    # 거리 관련

    df['haversine_dist']= haversine_array(df['lat'], df['long']) # 중심가로부터의 거리 (haversine_dist)

    df['haversine_dist_low'] = haversine_array_low(df['lat'], df['long'])

    #df['min_haver_multi'] = df['min_haver'] * df['haversine_dist'] 



# 로그화

for i in ['sqft_living', 'sqft_above', 'sqft_basement', 'sqft_living15', 'sqft_total_size', 'sqft_bedrooms'] :

    X_train[i] = np.log1p(X_train[i])

    X_test[i] = np.log1p(X_test[i])

    

X_train = X_train.drop([13522, 4123],0)

y_train = y_train.drop([13522, 4123],0)



X_train = X_train.reset_index(drop=True)

y_train = y_train.reset_index(drop=True)



print(len(X_train.columns), X_train.columns)

print([item for item in X_train.columns if item not in train_raw.columns])
X_train['price'] = y_train

km2 = KMeans(n_clusters=120, random_state=2019)

km2.fit(X_train[['price', 'km_120']])

X_train['km_pre'] = km2.predict(X_train[['price', 'km_120']])

X_train.drop('price', 1, inplace=True)

X_train_km_pre = X_train[['km_120', 'km_pre']].groupby('km_120')['km_pre'].mean().reset_index()

X_test = X_test.merge(X_train_km_pre, how='left', on='km_120')
plt.figure(figsize=(10,7))

plt.scatter(data['long'], data['lat'], s=5, c=np.log1p(data['price']), cmap='coolwarm')

plt.yticks(np.arange(47, 47.81, 0.05))

plt.xticks(np.arange(-122.6, -121.2, 0.15))

plt.grid(color='#BDBDBD', linestyle='-', linewidth=0.5)

plt.colorbar()

plt.scatter(-122.22, 47.63, color='green')

plt.scatter(-122.247, 47.382, color='purple')
X_train['price'] = y_train

f, ax =  plt.subplots(2,2, figsize=(20, 10))

sns.boxplot(x='sum_level', y='price', data=X_train, ax=ax[0,0])

sns.boxplot(x='sum_level_2', y='price', data=X_train, ax=ax[0,1])

sns.boxplot(x='bathrooms', y='price', data=X_train, ax=ax[1,0])

sns.boxplot(x='low_all', y='price', data=X_train, ax=ax[1,1])
plt.figure(figsize=(10,7))

plt.scatter(X_train['long'], X_train['lat'], s=5, c=X_train['km_120'])

plt.yticks(np.arange(47, 47.81, 0.05))

plt.xticks(np.arange(-122.6, -121.2, 0.15))

plt.grid(color='#BDBDBD', linestyle='-', linewidth=0.5)

plt.colorbar()
X_train.drop('price', 1, inplace=True)
%%time

dtrain = lgb.Dataset(X_train, label=y_train)

dtest  = lgb.Dataset(X_test)



lgb_params = {

    'boosting_type': 'gbdt',

    'objective':'regression',

    'num_leave' : 1,

    'learning_rate' : 0.03,

    'max_depth' : 6,

    'colsample_bytree' : 0.4,

    'subsample' : 0.4,

    'max_bin' : 80,

    'gpu_id':0,         

    'tree_method':'gpu_hist',

    'predictor':'gpu_predictor',

    'refit':True,

    'metric' : 'rmse',

    'seed' : 2019

}



cv_lgb_output = lgb.cv(lgb_params, dtrain, num_boost_round=5000, nfold=5, early_stopping_rounds=200, verbose_eval=100,stratified=False)



print('best_num_rounds :',len(cv_lgb_output['rmse-mean']))

print('best_cv_score :', cv_lgb_output['rmse-mean'][-1])



best_num_rounds = len(cv_lgb_output['rmse-mean'])



model_lgb = lgb.train(lgb_params, dtrain, num_boost_round=best_num_rounds)

lgb_pred_log = model_lgb.predict(X_test)

lgb_pred = np.expm1(lgb_pred_log)

xgb_params = {

    'eta': 0.02,

    'max_depth': 6,

    'subsample': 0.8,

    'colsample_bytree': 0.4,

    'objective': 'reg:linear',    

    'eval_metric': 'rmse',        

    'silent': True,               

    'seed' : 1984

}







# transform

dtrain = xgb.DMatrix(X_train, y_train)

dtest = xgb.DMatrix(X_test)



# cross validation

cv_output = xgb.cv(xgb_params,

                   dtrain,                        

                   num_boost_round=5000,         

                   early_stopping_rounds=150,    

                   nfold=5,                      

                   verbose_eval=100,             

                   feval=rmse_exp,               

                   maximize=False,

                   show_stdv=False,              

                   )



# scoring

best_rounds = cv_output.index.size

score = round(cv_output.iloc[-1]['test-rmse_exp-mean'], 2)



print(f'\nBest Rounds: {best_rounds}')

print(f'Best Score: {score}')



model = xgb.train(xgb_params, dtrain, num_boost_round=best_rounds)

xgb_pred_log = model.predict(dtest)

xgb_pred = np.expm1(xgb_pred_log)



# plotting

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,5))

cv_output[['train-rmse-mean', 'test-rmse-mean']].plot(ax=ax1)

ax1.set_title('RMSE_log', fontsize=20)

cv_output[['train-rmse_exp-mean', 'test-rmse_exp-mean']].plot(ax=ax2)

ax2.set_title('RMSE', fontsize=20)



plt.show()

ensemble_pred = np.vstack([lgb_pred, xgb_pred]).mean(0)

sample_submission = pd.read_csv('../input/sample_submission.csv')

submission = pd.DataFrame(data = {'id': test['id'], 'price': ensemble_pred})

submission.to_csv('submission_ensemble.csv', index=False)