# Loading packages

import pandas as pd #Analysis 

import matplotlib.pyplot as plt #Visulization

import seaborn as sns #Visulization

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)



import numpy as np #Analysis 

from scipy.stats import norm #Analysis 

from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer #Analysis 

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import roc_auc_score, roc_curve, mean_squared_error

from sklearn.cluster import KMeans

from scipy.special import boxcox1p

import xgboost as xgb

import lightgbm as lgb



import warnings 

warnings.filterwarnings('ignore')

import gc



import os

print(os.listdir("../input/2019-2nd-ml-month-with-kakr/"))



# Any results you write to the current directory are saved as output.
def pre_df(train_df, test_df):

    train_df = train_df.drop(['id'], axis=1)

    test_df = test_df.drop(['id'], axis=1)

    

    #target

    train_df['log_price'] = np.log(train_df.price)

    

    #date 

    train_df['date'] = train_df.date.map(lambda x : x[:6])

    test_df['date'] = test_df.date.map(lambda x : x[:6])

    train_df.date = train_df.date.astype(int)

    test_df.date = test_df.date.astype(int)

    

    #delete sqft_lot15

    train_df = train_df.drop(['sqft_lot15'], axis=1)

    test_df = test_df.drop(['sqft_lot15'], axis=1)



    #zip_level

    a = train_df[['zipcode', 'price']].groupby('zipcode').mean() 

    label = [j+1 for j in range(27)]

    a['zip_level'] = pd.cut(a.price, bins=27, labels=label)

    a = a.drop(['price'], axis=1)

    train_df = train_df.merge(a, on='zipcode', how='left')

    test_df = test_df.merge(a, on='zipcode', how='left')

    train_df.zip_level = train_df.zip_level.astype(int)

    test_df.zip_level = test_df.zip_level.astype(int)



    #zip_mean_price with Kmeans

    train_df['coord_cluster'] = None

    test_df['coord_cluster'] = None

    for i in train_df.zipcode.unique():

        df = train_df.loc[train_df.zipcode == i]

        coord = df[['lat','long']]

        num = (np.ceil(len(df) / 15)).astype(int)

        kmeans = KMeans(n_clusters=num, random_state=125).fit(coord)

        coord_cluster = kmeans.predict(coord)

        df['coord_cluster'] = coord_cluster

        df['coord_cluster'] = df['coord_cluster'].map(lambda x: 'c_' + str(x).rjust(2, '0'))

        train_df.loc[df.index, 'coord_cluster'] = df['coord_cluster']



        t_df = test_df.loc[test_df.zipcode == i]

        t_coord = t_df[['lat','long']]

        coord_cluster = kmeans.predict(t_coord)

        t_df['coord_cluster'] = coord_cluster

        t_df['coord_cluster'] = t_df['coord_cluster'].map(lambda x: 'c_' + str(x).rjust(2, '0'))

        test_df.loc[t_df.index, 'coord_cluster'] = t_df['coord_cluster']



    train_df['test'] = train_df['zipcode'].astype(str) + train_df['coord_cluster']

    test_df['test'] = test_df['zipcode'].astype(str) + test_df['coord_cluster']

    k = train_df[['price','test']].groupby('test').mean()

    k = k.rename(columns={'price' : 'mean_price'})

    train_df = pd.merge(train_df, k, how='left', on='test')

    test_df = pd.merge(test_df, k, how='left', on='test')

    train_df = train_df.rename(columns={'price_x' : 'price', 'price_y' : 'mean_price'})

    train_df = train_df.drop(['coord_cluster', 'test'], axis=1)

    test_df = test_df.drop(['coord_cluster', 'test'], axis=1)

    

    #is_re

    train_df['is_re'] = 0

    train_df.loc[train_df.loc[train_df.yr_renovated != 0].index, 'is_re'] = 1

    train_df.loc[train_df.loc[train_df.yr_renovated == 0].index, 'is_re'] = 0

    test_df['is_re'] = 0

    test_df.loc[test_df.loc[test_df.yr_renovated != 0].index, 'is_re'] = 1

    test_df.loc[test_df.loc[test_df.yr_renovated == 0].index, 'is_re'] = 0



    #yr_built

    train_df.loc[train_df.loc[train_df.yr_built < train_df.yr_renovated].index, 'yr_built'] = train_df.loc[

        train_df.yr_built < train_df.yr_renovated, 'yr_renovated']

    test_df.loc[test_df.loc[test_df.yr_built < test_df.yr_renovated].index, 'yr_built'] = test_df.loc[

        test_df.yr_built < test_df.yr_renovated, 'yr_renovated']



    #is_ba

    train_df['is_ba'] = 0

    train_df.loc[train_df.loc[train_df.sqft_basement != 0].index, 'is_ba'] = 1

    train_df.loc[train_df.loc[train_df.sqft_basement == 0].index, 'is_ba'] = 0

    test_df['is_ba'] = 0

    test_df.loc[test_df.loc[test_df.sqft_basement != 0].index, 'is_ba'] = 1

    test_df.loc[test_df.loc[test_df.sqft_basement == 0].index, 'is_ba'] = 0



    train_df = train_df.drop(['sqft_basement','yr_renovated'], axis=1)

    test_df = test_df.drop(['sqft_basement','yr_renovated'], axis=1)

    

    #living_rate

    train_df['living_rate'] = train_df['sqft_living'] / train_df['sqft_lot']

    test_df['living_rate'] = test_df['sqft_living'] / test_df['sqft_lot']

    

    #new_grade

    train_df['new_grade'] = train_df['grade'] + train_df['condition'] + train_df['view']

    test_df['new_grade'] = test_df['grade'] + test_df['condition'] + test_df['view']

    

    #per_living

    train_df['per_living'] = 0

    test_df['per_living'] = 0

    for i in train_df.zipcode.unique():

        tr_df = train_df.loc[train_df.zipcode == i]

        num = len(tr_df)

        te_df = test_df.loc[test_df.zipcode == i]

        df = pd.concat([tr_df, te_df], axis=0)



        min = df.sqft_living.min()

        max = df.sqft_living.max()

        ran = max-min



        df['per_living'] = (df['sqft_living'] - min) /ran

        df['k'] = pd.cut(df.per_living, bins=10, labels=[0,1,2,3,4,5,6,7,8,9])

        df['k'] = df['k'].astype(int)



        train_df.loc[tr_df.index, 'per_living'] = df['k'][:num]

        test_df.loc[te_df.index, 'per_living'] = df['k'][num:]

    print('finish')

    return train_df, test_df

print('data preparation function')
def zip_onehot(train_data, test_data):

    dummy = pd.get_dummies(train_data.zipcode, prefix='zipcode')

    train_data = pd.concat([train_data, dummy], axis=1)



    dummy = pd.get_dummies(test_data.zipcode, prefix='zipcode')

    test_data = pd.concat([test_data, dummy], axis=1)



    train_data = train_data.drop(['zipcode'], axis=1)

    test_data = test_data.drop(['zipcode'], axis=1)

    return train_data, test_data

#상위권 zipcode : 98118, 98033, 98006, 98122, 98112, 98106, 98108

print('one-hot encoder function')
def result(train_data):

    train_df = pd.read_csv('../input/2019-2nd-ml-month-with-kakr/train.csv')

    target = np.log(train_df.price)

    high_train = train_data.loc[train_df.price >= train_df.price.quantile(0.995)]

    h_target = target.loc[high_train.index]

    

    dtest = xgb.DMatrix(high_train)

    y_pred = model.predict(dtest)

    print('high_data error : ', np.sqrt(np.mean(np.square(np.exp(h_target) - np.exp(y_pred)))))



    dtest = xgb.DMatrix(train_data)

    y_pred = model.predict(dtest)

    print('all_data error : ', np.sqrt(np.mean(np.square(np.exp(target) - np.exp(y_pred)))))

print('check result function')
#xgb_params

def rmse_exp(predictions, dmat):

    labels = dmat.get_label()

    error = np.expm1(predictions) - np.expm1(labels)

    squared_error = np.square(error)

    mean = np.mean(squared_error)

    return ('rmse_exp', np.sqrt(mean))



xgb_params = {

    'eta': 0.02,

    'max_depth': 6,

    'subsample': 0.8,

    'colsample_bytree': 0.4,

#     'tree_method': 'gpu_hist',    # 최적화된 분할 지점을 찾기 위한 algorithm 설정 + 캐글의 GPU 사용

#     'predictor': 'gpu_predictor', # 예측 시에도 GPU사용

    'objective': 'reg:linear',    # 회귀

    'eval_metric': 'rmse',        # kaggle에서 요구하는 검증모델

    'silent': True,               # 학습 동안 메세지 출력할지 말지

    'seed': 4777,

}

print('xgb params')
#data load

train_df = pd.read_csv('../input/2019-2nd-ml-month-with-kakr/train.csv')

test_df = pd.read_csv('../input/2019-2nd-ml-month-with-kakr/test.csv')

#create result dataframe

test_result = pd.DataFrame(data={'id':test_df.id})

train_result = pd.DataFrame(data={'id':train_df.id})
a = train_df[['zipcode', 'price']].groupby('zipcode').mean()

a.head()
#zip_level

a = train_df[['zipcode', 'price']].groupby('zipcode').mean()

label = [j+1 for j in range(27)]

a['zip_level'] = pd.cut(a.price, bins=27, labels=label)

a = a.drop(['price'], axis=1)



train_df = train_df.merge(a, on='zipcode', how='left')

test_df = test_df.merge(a, on='zipcode', how='left')

train_df.zip_level = train_df.zip_level.astype(int)

test_df.zip_level = test_df.zip_level.astype(int)
print(train_df.price.corr(train_df.zip_level))

fig = plt.figure(figsize=(10, 6))

sns.boxplot(train_df.zip_level, train_df.price)
#zip_mean_price with Kmeans

train_df['coord_cluster'] = None

test_df['coord_cluster'] = None



for i in train_df.zipcode.unique():

    df = train_df.loc[train_df.zipcode == i]

    coord = df[['lat','long']]

    num = (np.ceil(len(df) / 15)).astype(int)

    kmeans = KMeans(n_clusters=num, random_state=125).fit(coord)

    coord_cluster = kmeans.predict(coord)

    df['coord_cluster'] = coord_cluster

    df['coord_cluster'] = df['coord_cluster'].map(lambda x: 'c_' + str(x).rjust(2, '0'))

    train_df.loc[df.index, 'coord_cluster'] = df['coord_cluster']



    t_df = test_df.loc[test_df.zipcode == i]

    t_coord = t_df[['lat','long']]

    coord_cluster = kmeans.predict(t_coord)

    t_df['coord_cluster'] = coord_cluster

    t_df['coord_cluster'] = t_df['coord_cluster'].map(lambda x: 'c_' + str(x).rjust(2, '0'))

    test_df.loc[t_df.index, 'coord_cluster'] = t_df['coord_cluster']



train_df['test'] = train_df['zipcode'].astype(str) + train_df['coord_cluster']

test_df['test'] = test_df['zipcode'].astype(str) + test_df['coord_cluster']

k = train_df[['price','test']].groupby('test').mean()

k = k.rename(columns={'price' : 'mean_price'})



train_df = pd.merge(train_df, k, how='left', on='test')

test_df = pd.merge(test_df, k, how='left', on='test')

train_df = train_df.rename(columns={'price_x' : 'price', 'price_y' : 'mean_price'})



train_df = train_df.drop(['coord_cluster', 'test'], axis=1)

test_df = test_df.drop(['coord_cluster', 'test'], axis=1)
fig = plt.figure(figsize=(12,8))

sns.scatterplot(x='long',y='lat',hue='mean_price',size='price',sizes=(5,100), data=train_df)

print(train_df.price.corr(train_df.mean_price))
#is_re

train_df['is_re'] = 0

train_df.loc[train_df.loc[train_df.yr_renovated != 0].index, 'is_re'] = 1

train_df.loc[train_df.loc[train_df.yr_renovated == 0].index, 'is_re'] = 0



test_df['is_re'] = 0

test_df.loc[test_df.loc[test_df.yr_renovated != 0].index, 'is_re'] = 1

test_df.loc[test_df.loc[test_df.yr_renovated == 0].index, 'is_re'] = 0



#yr_built

train_df.loc[train_df.loc[train_df.yr_built < train_df.yr_renovated].index, 'yr_built'] = train_df.loc[

    train_df.yr_built < train_df.yr_renovated, 'yr_renovated']



test_df.loc[test_df.loc[test_df.yr_built < test_df.yr_renovated].index, 'yr_built'] = test_df.loc[

    test_df.yr_built < test_df.yr_renovated, 'yr_renovated']



#is_ba

train_df['is_ba'] = 0

train_df.loc[train_df.loc[train_df.sqft_basement != 0].index, 'is_ba'] = 1

train_df.loc[train_df.loc[train_df.sqft_basement == 0].index, 'is_ba'] = 0



test_df['is_ba'] = 0

test_df.loc[test_df.loc[test_df.sqft_basement != 0].index, 'is_ba'] = 1

test_df.loc[test_df.loc[test_df.sqft_basement == 0].index, 'is_ba'] = 0



train_df = train_df.drop(['sqft_basement','yr_renovated'], axis=1)

test_df = test_df.drop(['sqft_basement','yr_renovated'], axis=1)
train_df['living_rate'] = train_df['sqft_living'] / train_df['sqft_lot']

test_df['living_rate'] = test_df['sqft_living'] / test_df['sqft_lot']
train_df['new_grade'] = train_df['grade'] + train_df['condition'] + train_df['view']

test_df['new_grade'] = test_df['grade'] + test_df['condition'] + test_df['view']
train_df['per_living'] = 0

test_df['per_living'] = 0

for i in train_df.zipcode.unique():

    tr_df = train_df.loc[train_df.zipcode == i]

    num = len(tr_df)

    te_df = test_df.loc[test_df.zipcode == i]

    df = pd.concat([tr_df, te_df], axis=0)



    min = df.sqft_living.min()

    max = df.sqft_living.max()

    ran = max-min



    df['per_living'] = (df['sqft_living'] - min) /ran

    df['k'] = pd.cut(df.per_living, bins=10, labels=[0,1,2,3,4,5,6,7,8,9])

    df['k'] = df['k'].astype(int)



    train_df.loc[tr_df.index, 'per_living'] = df['k'][:num]

    test_df.loc[te_df.index, 'per_living'] = df['k'][num:]
def skew(train_data, test_data, ca_var):

    skewness = train_data[ca_var].skew()

    skew_var = skewness.index[skewness.values >= 0.5]



    lam = 0.15

    for var in skew_var:

        #c = 'box_' + var

        train_data[var] = boxcox1p(train_data[var], lam)

        test_data[var] = boxcox1p(test_data[var], lam)

        #train_data[var] = np.log1p(train_data[var])

        #test_data[var] = np.log1p(test_data[var])

    print(skew_var)    

    return train_data, test_data
ca_var = ['sqft_living', 'sqft_lot', 'sqft_above', 'sqft_living15','mean_price']

train_df, test_df = skew(train_df, test_df, ca_var)
train_df = pd.read_csv('../input/2019-2nd-ml-month-with-kakr/train.csv')

test_df = pd.read_csv('../input/2019-2nd-ml-month-with-kakr/test.csv')
train_data, test_data = pre_df(train_df, test_df)

ca_var = ['sqft_living', 'sqft_lot', 'sqft_above', 'sqft_living15', 'mean_price']

train_data, test_data = skew(train_data, test_data, ca_var)

train_data, test_data = zip_onehot(train_data, test_data)



target = train_data.log_price

train_data = train_data.drop(['price', 'log_price'], axis=1)

print(len(train_data.columns), len(test_data.columns))
%%time

# transforming

dtrain = xgb.DMatrix(train_data, target)

dtest = xgb.DMatrix(test_data)



# cross validation

cv_output = xgb.cv(xgb_params,

                   dtrain,                        

                   num_boost_round=5000,         # the number of boosting trees

                   early_stopping_rounds=100,    # val loss가 계속 상승하면 중지

                   nfold=5,                      # set folds of the closs validation

                   verbose_eval=250,             # 몇 번째마다 메세지를 출력할 것인지

                   feval=rmse_exp,               # price 속성을 log scaling 했기 때문에, 다시 exponential

                   maximize=False,

                   show_stdv=False,              # 학습 동안 std(표준편차) 출력할지 말지

                   )



# scoring

best_rounds = cv_output.index.size

score = round(cv_output.iloc[-1]['test-rmse_exp-mean'], 2)



print(f'\nBest Rounds: {best_rounds}')

print(f'Best Score: {score}')
#1번

model = xgb.train(xgb_params, dtrain, num_boost_round=best_rounds)



dtest = xgb.DMatrix(test_data)

y_pred = model.predict(dtest)

test_result['pred_1'] = np.exp(y_pred)



result(train_data)

dtest = xgb.DMatrix(train_data)

train_result['pred_1'] = np.exp(model.predict(dtest))
train_df = pd.read_csv('../input/sample-code-with-2019-ml-month-2nd/train_df_2.csv')

test_df = pd.read_csv('../input/sample-code-with-2019-ml-month-2nd/test_df_2.csv')
train_df = train_df.drop(['alone','dist_arr', 'p_living','p_above','p_lot', 'p_mean','l_mean'], axis=1)

test_df = test_df.drop(['c_index','over','p_mean', 'l_mean'], axis=1)
train_data, test_data = pre_df(train_df, test_df)

ca_var = ['sqft_living', 'sqft_lot', 'sqft_above', 'sqft_living15', 'mean_price', 'a_mean']

train_data, test_data = skew(train_data, test_data, ca_var)

train_data, test_data = zip_onehot(train_data, test_data)



target = train_data.log_price

train_data = train_data.drop(['price','log_price'], axis=1)

print(len(train_data.columns), len(test_data.columns))
%%time

# transforming

dtrain = xgb.DMatrix(train_data, target)

dtest = xgb.DMatrix(test_data)



# cross validation

cv_output = xgb.cv(xgb_params,

                   dtrain,                        

                   num_boost_round=5000,         # the number of boosting trees

                   early_stopping_rounds=100,    # val loss가 계속 상승하면 중지

                   nfold=5,                      # set folds of the closs validation

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
#2번

model = xgb.train(xgb_params, dtrain, num_boost_round=best_rounds)

dtest = xgb.DMatrix(test_data)

y_pred = model.predict(dtest)

test_result['pred_2'] = np.exp(y_pred)



result(train_data)

dtest = xgb.DMatrix(train_data)

train_result['pred_2'] = np.exp(model.predict(dtest))
train_df = pd.read_csv('../input/sample-code-with-2019-ml-month-2nd/train_df_2.csv')

test_df = pd.read_csv('../input/sample-code-with-2019-ml-month-2nd/test_df_2.csv')
train_df = train_df.drop(['alone','dist_arr', 'p_living','p_above', 'p_lot'], axis=1)

test_df = test_df.drop(['c_index','over'], axis=1)
train_data, test_data = pre_df(train_df, test_df)

ca_var = ['sqft_living', 'sqft_lot', 'sqft_above', 'sqft_living15', 'mean_price', 'p_mean', 'l_mean', 'a_mean']

train_data, test_data = skew(train_data, test_data, ca_var)

train_data, test_data = zip_onehot(train_data, test_data)



target = train_data.log_price

train_data = train_data.drop(['price','log_price'], axis=1)

print(len(train_data.columns), len(test_data.columns))
%%time

# transforming

dtrain = xgb.DMatrix(train_data, target)

dtest = xgb.DMatrix(test_data)



# cross validation

cv_output = xgb.cv(xgb_params,

                   dtrain,                        

                   num_boost_round=5000,         # the number of boosting trees

                   early_stopping_rounds=100,    # val loss가 계속 상승하면 중지

                   nfold=5,                      # set folds of the closs validation

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
#2번

model = xgb.train(xgb_params, dtrain, num_boost_round=best_rounds)

dtest = xgb.DMatrix(test_data)

y_pred = model.predict(dtest)

test_result['pred_6'] = np.exp(y_pred)



result(train_data)

dtest = xgb.DMatrix(train_data)

train_result['pred_6'] = np.exp(model.predict(dtest))
train_df = pd.read_csv('../input/dataset/train_data_2.csv')

test_df = pd.read_csv('../input/dataset/test_data_2.csv')
train_data = train_df.copy()

test_data = test_df.copy()

train_data, test_data = zip_onehot(train_data, test_data)
%%time

# transforming

dtrain = xgb.DMatrix(train_data, target)

dtest = xgb.DMatrix(test_data)



# cross validation

cv_output = xgb.cv(xgb_params,

                   dtrain,                        

                   num_boost_round=5000,         # the number of boosting trees

                   early_stopping_rounds=100,    # val loss가 계속 상승하면 중지

                   nfold=5,                      # set folds of the closs validation

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
#3번

model = xgb.train(xgb_params, dtrain, num_boost_round=best_rounds)

dtest = xgb.DMatrix(test_data)

y_pred = model.predict(dtest)

test_result['pred_3'] = np.exp(y_pred)



result(train_data)

dtest = xgb.DMatrix(train_data)

train_result['pred_3'] = np.exp(model.predict(dtest))
def feature(x):

    x['whether_to_renovated']=(x.yr_renovated!=0).astype(int)

    x.loc[x.yr_renovated==0,'yr_renovated']=x[x.yr_renovated==0].yr_built

    x=pd.DataFrame.drop(x,columns='yr_built')

    x['garret']=(x.floors%1==0.5).astype(int)

    x.loc[x.floors%1==0.5,'floors']=np.floor(x[x.floors%1==0.5].floors)

    

#     x['rooms_mul']=x['bedrooms']*x['bathrooms']

    x['living_per_floors']=x['sqft_living']/x['floors']

    x['total_score']=x['condition']+x['grade']+x['view']

    x['living_per_lot']=x['sqft_living']/x['sqft_lot']

    x['diff_of_rooms']=np.abs(x['bedrooms']-x['bathrooms'])

    x['diff_lots']=np.abs(x['sqft_lot15']-x['sqft_lot'])

    x['diff_living']=np.abs(x['sqft_living15']-x['sqft_living'])

    x['diff_living_per_floor']=(x.sqft_living15-x.sqft_living)/x.floors

    x['exist_special']=x.garret+x.waterfront+x.whether_to_renovated

#     x['where_water']=np.abs(x.long*x.lat*x.waterfront)

#     x['lat*lot']=x.lat*x.sqft_lot 9.9e4

#     x['lat*long']=np.abs(x.lat*x.long)

#     x['base*above']=x.sqft_basement*x.sqft_above

#     x['total*score']=x.condition*x.grade*x.view

    return x



def datererange(x):

    x.loc[:,'date']=x.loc[:,'date'].str[:6].astype(int)

    index=np.sort(x.date.unique())

    for i in range(len(index)):

        x.loc[x.date==index[i],'date']=i+1

    return x

    

def log1pscale(x,cols):

    for i in cols:

        x.loc[:,i]=np.log1p(x.loc[:,i])

    return x



def logscale(x,cols):

    for i in cols:

        x.loc[:,i]=np.log(x.loc[:,i])

    return x



def seed():

    return np.random.randint(10000)



data=pd.read_csv('../input/2019-2nd-ml-month-with-kakr/train.csv').drop(columns='id')

test=pd.read_csv('../input/2019-2nd-ml-month-with-kakr/test.csv').drop(columns='id')



target=data.price

log_target=np.log(target)

data=data.drop(columns='price')



raw=data.copy()



data=feature(data)

test=feature(test)



log1p_cols=['sqft_living','sqft_living15','sqft_lot','sqft_lot15','sqft_basement','living_per_floors',

           'diff_lots','diff_living',]

data=log1pscale(data,log1p_cols)

test=log1pscale(test,log1p_cols)

data=datererange(data)

test=datererange(test)



data.describe()



eliminate_list=['price','sqft_lot15']

for i in eliminate_list:

    if i in data.columns:

        data=data.drop(columns=i)

    if i in test.columns:

        if i != 'price':

            test=test.drop(columns=i)

            

x_sample,x_unseen,y_sample,y_unseen=train_test_split(data,log_target,test_size=1/5)

watchlist=[(x_sample,y_sample),(x_unseen,y_unseen)]





modelx=xgb.XGBRegressor(tree_method='gpu_hist',

                        n_estimators=100000,

                        num_round_boost=500,

                        show_stdv=False,

                        feature_selector='greedy',

                        verbosity=0,

                        reg_lambda=10,

                        reg_alpha=0.01,

                        learning_rate=0.001,

                        seed=seed(),

                        colsample_bytree=0.8,

                        colsample_bylevel=0.8,

                        subsample=0.8,

                        n_jobs=-1,

                        gamma=0.005,

                        base_score=np.mean(log_target)

                       )



modelx.fit(x_sample,y_sample,verbose=False,eval_set=watchlist,

             eval_metric='rmse',

          early_stopping_rounds=1000)
from sklearn.metrics import mean_squared_error as mse

#xgb_score=mse(np.exp(modelx.predict(x_unseen)),np.exp(y_unseen))**0.5



xgb_train_pred=np.exp(modelx.predict(data))

xgb_pred=np.exp(modelx.predict(test))





print("RMSE unseen : {}".format(\

        mse(np.exp(modelx.predict(x_unseen)),np.exp(y_unseen))**0.5))





fig, ax = plt.subplots(figsize=(10,10))

xgb.plot_importance(modelx, ax=ax)

plt.show()
#4번 : https://www.kaggle.com/marchen911/xgboost-lightgbm/notebook 코드

test_result['pred_4'] = pd.DataFrame(xgb_pred)

train_result['pred_4'] = pd.DataFrame(xgb_train_pred)
train_df = pd.read_csv('../input/2019-2nd-ml-month-with-kakr/train.csv')

test_df = pd.read_csv('../input/2019-2nd-ml-month-with-kakr/test.csv')
train_data, test_data = pre_df(train_df, test_df)

train_data = pd.concat([train_data, train_result], axis=1)



test_data = pd.concat([test_data, test_result], axis=1)

test_data = test_data.drop(['id'], axis=1)
ca_var = ['sqft_living', 'sqft_lot', 'sqft_above','sqft_living15','mean_price',  'pred_1', 'pred_2', 'pred_3', 'pred_4', 'pred_6']

train_data, test_data = skew(train_data, test_data, ca_var)

train_data, test_data = zip_onehot(train_data, test_data)



target = train_data.log_price

train_data = train_data.drop(['id', 'price','log_price'], axis=1)

print(len(train_data.columns), len(test_data.columns))
%%time

# transforming

dtrain = xgb.DMatrix(train_data, target)

dtest = xgb.DMatrix(test_data)



# cross validation

cv_output = xgb.cv(xgb_params,

                   dtrain,                        

                   num_boost_round=5000,         # the number of boosting trees

                   early_stopping_rounds=100,    # val loss가 계속 상승하면 중지

                   nfold=5,                      # set folds of the closs validation

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
#5번

model = xgb.train(xgb_params, dtrain, num_boost_round=best_rounds)

dtest = xgb.DMatrix(test_data)

y_pred = model.predict(dtest)

test_result['pred_5'] = np.exp(y_pred)



result(train_data)

dtest = xgb.DMatrix(train_data)

train_result['pred_5'] = np.exp(model.predict(dtest))
from scipy.cluster.hierarchy import dendrogram, linkage  

from scipy.spatial.distance import  pdist

from matplotlib import pyplot as plt

from sklearn.preprocessing import MinMaxScaler



# system

from datetime import datetime

import os

sns.set()
solutions_set = test_result[['pred_1','pred_2','pred_3','pred_4','pred_5','pred_6']]
solutions_set.head()
# Scaling

scaler = MinMaxScaler()  

solutions_set_scaled = scaler.fit_transform(solutions_set)

solutions_set_scaled = pd.DataFrame(solutions_set_scaled, columns = solutions_set.columns)
# transpose and convert solutions set to numpy

np_solutions_set = solutions_set_scaled.T.values

# calculate the distances

solutions_set_dist = pdist(np_solutions_set)

# hierarchical clusterization

linked = linkage(solutions_set_dist, 'ward')



# dendrogram

fig = plt.figure(figsize=(8, 5))

dendrogram(linked, labels = solutions_set_scaled.columns)

plt.title('clusters')

plt.show()
c1 = (solutions_set['pred_5'] + solutions_set['pred_1'] + solutions_set['pred_3'] + solutions_set['pred_4']) / 4

c2 = (solutions_set['pred_2'] + solutions_set['pred_6'])/2

pred = c1 * 0.55+c2 * 0.45
#pred

tt = pd.read_csv('../input/2019-2nd-ml-month-with-kakr/test.csv')

sub_id = tt.id

sub = pd.DataFrame(data={'id':sub_id,'price':pred})
sub.to_csv('submission.csv', index=False)