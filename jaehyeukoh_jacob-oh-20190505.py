# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Loading packages

import pandas as pd #Analysis 

import matplotlib.pyplot as plt #Visulization

import seaborn as sns #Visulization

import numpy as np #Analysis 

from scipy.stats import norm #Analysis 

from sklearn.preprocessing import StandardScaler #Analysis 

from scipy import stats #Analysis 

import warnings 

warnings.filterwarnings('ignore')

%matplotlib inline

import gc



import lightgbm as lgb

import xgboost as xgb

import catboost as cb



import math

import time

from datetime import datetime, timedelta,date



from sklearn.model_selection import StratifiedKFold, KFold

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import ElasticNet, Lasso, Ridge

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.svm import SVR

from sklearn.preprocessing import RobustScaler
def check_train_test_diff(train, test, col):

    ratio_df = pd.concat([train[col].value_counts()/train.shape[0], test[col].value_counts()/test.shape[0]], axis=1)

    ratio_df.columns = ['train','test']

    ratio_df['diff'] = ratio_df['train'] - ratio_df['test']

    return ratio_df
def category_feature_distribution(train, col, target='price'):

    fig, ax = plt.subplots(1, 2, figsize=(16,4))

    

    for c in sorted(train[col].unique()):

        sns.distplot(np.log1p(train.loc[train[col]==c, target]), ax=ax[0])

    ax[0].legend(sorted(train[col].unique()))

    ax[0].set_title(f'{col} {target} distribution')



    sns.boxplot(x=col, y=target, data=df_train, ax=ax[1])

    ax[1].set_title(f'{col} vs {target}')

    

    plt.show()
def continous_feature_distribution(train, test, col, target='price'):

    fig, ax = plt.subplots(1, 2, figsize=(12,5))

    

    sns.distplot(train[col], ax=ax[0])

    sns.distplot(test[col], ax=ax[0])

    ax[0].set_title(f'{col} - train/test distribution')

    

    sns.scatterplot(x=col, y=target, data=train, ax=ax[1])

    ax[1].set_title(f'{col} - {target} scatterplot')

    

    plt.show()
def scatter_quantile_graph(frame, col1, col2):

    col1_quantile = np.arange(0,1.1,0.1)

    col2_quantile = np.arange(0,1.1,0.1)



    for quantile_value in frame[col1].quantile(col1_quantile):

        plt.axvline(quantile_value, color='red', alpha=0.3)

    for quantile_value in frame[col2].quantile(col2_quantile):

        plt.axhline(quantile_value, color='blue', alpha=0.3)

        

    sns.scatterplot(col1, col2, hue='price',data=frame)

    

    plt.title('{} - {}'.format(col1,col2))

    plt.show()
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

    

def groupby_helper(df, group_col, target_col, agg_method, prefix_param=None):

    try:

        prefix = get_prefix(group_col, target_col, prefix_param)

        print(group_col, target_col, agg_method)

        group_df = df.groupby(group_col)[target_col].agg(agg_method)

        group_df.columns = ['{}_{}'.format(prefix, m) for m in agg_method]

    except BaseException as e:

        print(e)

    return group_df.reset_index()
from functools import wraps

def time_decorator(func): 

    @wraps(func)

    def wrapper(*args, **kwargs):

        print("\nStartTime: ", datetime.now() + timedelta(hours=9))

        start_time = time.time()

        

        df = func(*args, **kwargs)

        

        print("EndTime: ", datetime.now() + timedelta(hours=9))  

        print("TotalTime: ", time.time() - start_time)

        return df

        

    return wrapper



class SklearnWrapper(object):

    def __init__(self, clf, params=None, **kwargs):

        #if isinstance(SVR) is False:

        #    params['random_state'] = kwargs.get('seed', 0)

        self.clf = clf(**params)

        self.is_classification_problem = True

    @time_decorator

    def train(self, x_train, y_train, x_cross=None, y_cross=None):

        if len(np.unique(y_train)) > 30:

            self.is_classification_problem = False

            

        self.clf.fit(x_train, y_train)



    def predict(self, x):

        if self.is_classification_problem is True:

            return self.clf.predict_proba(x)[:,1]

        else:

            return self.clf.predict(x)   

    

class XgbWrapper(object):

    def __init__(self, params=None, **kwargs):

        self.param = params

        self.param['seed'] = kwargs.get('seed', 0)

        self.num_rounds = kwargs.get('num_rounds', 1000)

        self.early_stopping = kwargs.get('ealry_stopping', 100)



        self.eval_function = kwargs.get('eval_function', None)

        self.verbose_eval = kwargs.get('verbose_eval', 100)

        self.best_round = 0

    

    @time_decorator

    def train(self, x_train, y_train, x_cross=None, y_cross=None):

        need_cross_validation = True

       

        if isinstance(y_train, pd.DataFrame) is True:

            y_train = y_train[y_train.columns[0]]

            if y_cross is not None:

                y_cross = y_cross[y_cross.columns[0]]



        if x_cross is None:

            dtrain = xgb.DMatrix(x_train, label=y_train, silent= True)

            train_round = self.best_round

            if self.best_round == 0:

                train_round = self.num_rounds

            

            print(train_round)

            self.clf = xgb.train(self.param, dtrain, train_round)

            del dtrain

        else:

            dtrain = xgb.DMatrix(x_train, label=y_train, silent=True)

            dvalid = xgb.DMatrix(x_cross, label=y_cross, silent=True)

            watchlist = [(dtrain, 'train'), (dvalid, 'eval')]



            self.clf = xgb.train(self.param, dtrain, self.num_rounds, watchlist, feval=self.eval_function,

                                 early_stopping_rounds=self.early_stopping,

                                 verbose_eval=self.verbose_eval)

            self.best_round = max(self.best_round, self.clf.best_iteration)



    def predict(self, x):

        return self.clf.predict(xgb.DMatrix(x), ntree_limit=self.best_round)



    def get_params(self):

        return self.param    

    

class LgbmWrapper(object):

    def __init__(self, params=None, **kwargs):

        self.param = params

        self.param['seed'] = kwargs.get('seed', 0)

        self.num_rounds = kwargs.get('num_rounds', 1000)

        self.early_stopping = kwargs.get('ealry_stopping', 100)



        self.eval_function = kwargs.get('eval_function', None)

        self.verbose_eval = kwargs.get('verbose_eval', 100)

        self.best_round = 0

        

    @time_decorator

    def train(self, x_train, y_train, x_cross=None, y_cross=None):

        """

        x_cross or y_cross is None

        -> model train limted num_rounds

        

        x_cross and y_cross is Not None

        -> model train using validation set

        """

        if isinstance(y_train, pd.DataFrame) is True:

            y_train = y_train[y_train.columns[0]]

            if y_cross is not None:

                y_cross = y_cross[y_cross.columns[0]]



        if x_cross is None:

            dtrain = lgb.Dataset(x_train, label=y_train, silent= True)

            train_round = self.best_round

            if self.best_round == 0:

                train_round = self.num_rounds

                

            self.clf = lgb.train(self.param, train_set=dtrain, num_boost_round=train_round)

            del dtrain   

        else:

            dtrain = lgb.Dataset(x_train, label=y_train, silent=True)

            dvalid = lgb.Dataset(x_cross, label=y_cross, silent=True)

            self.clf = lgb.train(self.param, train_set=dtrain, num_boost_round=self.num_rounds, valid_sets=[dtrain, dvalid],

                                  feval=self.eval_function, early_stopping_rounds=self.early_stopping,

                                  verbose_eval=self.verbose_eval)

            self.best_round = max(self.best_round, self.clf.best_iteration)

            del dtrain, dvalid

            

        gc.collect()

    

    def predict(self, x):

        return self.clf.predict(x, num_iteration=self.clf.best_iteration)

    

    def plot_importance(self):

        lgb.plot_importance(self.clf, max_num_features=50, height=0.7, figsize=(10,30))

        plt.show()

        

    def get_params(self):

        return self.param

    

    

@time_decorator

def get_oof(clf, x_train, y_train, x_test, eval_func, **kwargs):

    nfolds = kwargs.get('NFOLDS', 5)

    kfold_shuffle = kwargs.get('kfold_shuffle', True)

    kfold_random_state = kwargs.get('kfold_random_state', 0)

    stratified_kfold_ytrain = kwargs.get('stratifed_kfold_y_value', None)

    ntrain = x_train.shape[0]

    ntest = x_test.shape[0]

    

    kf_split = None

    if stratified_kfold_ytrain is None:

        kf = KFold(n_splits=nfolds, shuffle=kfold_shuffle, random_state=kfold_random_state)

        kf_split = kf.split(x_train)

    else:

        kf = StratifiedKFold(n_splits=nfolds, shuffle=kfold_shuffle, random_state=kfold_random_state)

        kf_split = kf.split(x_train, stratified_kfold_ytrain)

        

    oof_train = np.zeros((ntrain,))

    oof_test = np.zeros((ntest,))



    cv_sum = 0

    

    # before running model, print model param

    # lightgbm model and xgboost model use get_params()

    try:

        if clf.clf is not None:

            print(clf.clf)

    except:

        print(clf)

        print(clf.get_params())



    for i, (train_index, cross_index) in enumerate(kf_split):

        x_tr, x_cr = None, None

        y_tr, y_cr = None, None

        if isinstance(x_train, pd.DataFrame):

            x_tr, x_cr = x_train.iloc[train_index], x_train.iloc[cross_index]

            y_tr, y_cr = y_train.iloc[train_index], y_train.iloc[cross_index]

        else:

            x_tr, x_cr = x_train[train_index], x_train[cross_index]

            y_tr, y_cr = y_train[train_index], y_train[cross_index]



        clf.train(x_tr, y_tr, x_cr, y_cr)

        

        oof_train[cross_index] = clf.predict(x_cr)



        cv_score = eval_func(y_cr, oof_train[cross_index])

        

        print('Fold %d / ' % (i+1), 'CV-Score: %.6f' % cv_score)

        cv_sum = cv_sum + cv_score

        

        del x_tr, x_cr, y_tr, y_cr

        

    gc.collect()

    

    score = cv_sum / nfolds

    print("Average CV-Score: ", score)



    # Using All Dataset, retrain

    clf.train(x_train, y_train)

    oof_test = clf.predict(x_test)



    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1), score
df_train = pd.read_csv('../input/train.csv')

df_test  = pd.read_csv('../input/test.csv')
print("train.csv. Shape: ",df_train.shape)

print("test.csv. Shape: ",df_test.shape)
default_check = pd.concat([df_train.isnull().sum(), df_train.dtypes, df_train.nunique(), df_train.describe().T], axis=1)

default_check.rename(columns={0:'NULL', 1:'TYPE', 2:'UNIQUE'}, inplace=True)

default_check
df_test.loc[df_test['bedrooms'] == 33, 'bedrooms'] = 3

df_train[df_train.bathrooms == 0]
fig, ax = plt.subplots(1, 3, figsize=(20,4))

sns.distplot(df_train['price'], ax=ax[0])

sns.distplot(np.log1p(df_train['price']), ax=ax[1])

df_train['price'].plot(ax=ax[2])

plt.show()
from IPython.display import display

for col in ['bedrooms', 'bathrooms', 'floors', 'waterfront', 'view', 'condition', 'grade']:

    print(col)

    display(check_train_test_diff(df_train, df_test, col))
df_train['date'] = pd.to_datetime(df_train['date'])

print(df_train['date'].min(), df_train['date'].max())



df_test['date'] = pd.to_datetime(df_test['date'])

print(df_test['date'].min(), df_test['date'].max())
print("Train")

display(df_train.sort_values('date').head())

print("Test")

display(df_test.sort_values('date').head())
plt.plot(df_train.sort_values('date')['price'].cumsum().values)
fig, ax = plt.subplots(1, 4, figsize=(20,5))

df_train.groupby('date')['price'].count().plot(ax=ax[0])

ax[0].set_title('Each date price count')

df_train.groupby('date')['price'].sum().plot(ax=ax[1])

ax[1].set_title('Each date price sum')

df_train.groupby('date')['price'].mean().plot(ax=ax[2])

ax[2].set_title('Each date price mean')

df_train.groupby('date')['price'].std().plot(ax=ax[3])

ax[3].set_title('Each date price std')

plt.show()
df_train.loc[df_train['date']==np.argmax(df_train.groupby('date')['price'].mean())]

fig, ax = plt.subplots(1, 2, figsize=(10,5))

df_train.groupby('date')['id'].count().plot(ax=ax[0])

ax[0].set_title('Train each date sales count')

df_test.groupby('date')['id'].count().plot(ax=ax[1])

ax[1].set_title('Test each date sales count')

plt.show()
df_train['yearmonth'] = df_train['date'].dt.year*100 + df_train['date'].dt.month

category_feature_distribution(df_train,'yearmonth')
area_feature = ['sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement', 'sqft_living15', 'sqft_lot15']

for col in area_feature:

    continous_feature_distribution(df_train, df_test, col)
sns.scatterplot('long','lat',hue='price',data=df_train)
scatter_quantile_graph(df_train, 'long', 'lat')
qcut_count = 20

df_train['qcut_long'] = pd.qcut(df_train['long'], qcut_count, labels=range(qcut_count))

df_train['qcut_lat'] = pd.qcut(df_train['lat'], qcut_count, labels=range(qcut_count))

df_test['qcut_long'] = pd.qcut(df_test['long'], qcut_count, labels=range(qcut_count))

df_test['qcut_lat'] = pd.qcut(df_test['lat'], qcut_count, labels=range(qcut_count))

temp = df_train.groupby(['qcut_long','qcut_lat'])['price'].mean().reset_index()

sns.scatterplot('qcut_long','qcut_lat', hue='price', data=temp);

# temp

df_train['qcut_long'], df_train['qcut_lat']
# df_train_q = df_train.merge(temp, left_on=['qcut_long', 'qcut_lat'], right_on=['qcut_long', 'qcut_lat'], suffixes=('_left', '_right'))

# df_test_q = df_test.merge(temp, left_on=['qcut_long', 'qcut_lat'], right_on=['qcut_long', 'qcut_lat'], suffixes=('_left', '_right'))

df_train_q = df_train.copy()

df_test_q = df_test.copy()

# temp.merge(df_train)
df_train_q.bedrooms[df_train_q.id == 4123] = 1

df_train_q.bathrooms[df_train_q.id == 4123] = 0.75

df_train_q.bedrooms[df_train_q.id == 6885] = 2

df_train_q.bathrooms[df_train_q.id == 6885] = 2.5

df_train_q.bedrooms[df_train_q.id == 7322] = 1

df_train_q.bathrooms[df_train_q.id == 7322] = 1

df_train_q.bedrooms[df_train_q.id == 8826] = 3

df_train_q.bathrooms[df_train_q.id == 8826] = 2.5

df_train_q.bedrooms[df_train_q.id == 12781] = 1

df_train_q.bathrooms[df_train_q.id == 12781] = 0.75

df_train_q = df_train_q.drop(df_train_q[(df_train_q['id']==13522)].index)
df_test_q.bedrooms[df_test_q.id == 15911] = 3

df_test_q.bathrooms[df_test_q.id == 15911] = 1.75

df_test_q.bedrooms[df_test_q.id == 19312] = 2

df_test_q.bathrooms[df_test_q.id == 19312] = 1

df_test_q.bedrooms[df_test_q.id == 17935] = 4

df_test_q.bathrooms[df_test_q.id == 17935] = 2.5

df_test_q.bedrooms[df_test_q.id == 15265] = 4

df_test_q.bathrooms[df_test_q.id == 15265] = 3

df_test_q.bedrooms[df_test_q.id == 15345] = 1

df_test_q.bathrooms[df_test_q.id == 15345] = 1

df_test_q.bedrooms[df_test_q.id == 17064] = 4

df_test_q.bathrooms[df_test_q.id == 17064] = 4.25

df_test_q.bedrooms[df_test_q.id == 17524] = 4

df_test_q.bathrooms[df_test_q.id == 17524] = 4.25

df_test_q.bedrooms[df_test_q.id == 17526] = 4

df_test_q.bathrooms[df_test_q.id == 17526] = 2.5

df_test_q.bedrooms[df_test_q.id == 16012] = 2

df_test_q.bathrooms[df_test_q.id == 16012] = 1.5

df_test_q.bedrooms[df_test_q.id == 16418] = 1

df_test_q.bathrooms[df_test_q.id == 16418] = 1

# df_train_q[df_train_q.bedrooms == 0][['id', 'price_left', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'grade', 'qcut_long', 'qcut_lat', 'price_right']]

df_test_q[df_test_q.bathrooms == 0][['id', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'grade', 'qcut_long', 'qcut_lat']]

df_test_q[~df_test_q.id.isin(df_test.id)]

df_test[~df_test.id.isin(df_test_q.id)]



df_train_q = df_train_q.loc[df_train_q['grade']>3]

df_train_q = df_train_q.loc[df_train_q['id']!=10350]

'''

df_train = df_train.loc[df_train['price'] <= 7000000]

df_train = df_train.loc[df_train['bedrooms']<=10]

df_train = df_train.loc[df_train['bathrooms']<7]

df_train = df_train.loc[df_train['grade']>1]

df_train = df_train.loc[df_train['id']!=13311]



df_train = df_train.drop(df_train[(df_train['grade']<=3)].index)

df_train = df_train.drop(df_train[(df_train['bedrooms']>=10)].index)

df_train = df_train.drop(df_train[(df_train['bathrooms']==6.25)].index)

df_train = df_train.drop(df_train[(df_train['bathrooms']==6.75)].index)

df_train = df_train.drop(df_train[(df_train['bathrooms']==7.5)].index)

df_train = df_train.drop(df_train[(df_train['bathrooms']==7.75)].index)

df_train = df_train.drop(df_train[(df_train['bathrooms']==4.5) & (df_train['price']>15)].index)

df_train = df_train.drop(df_train.loc[(df_train['sqft_living']>13000) & (df_train['bathrooms']>7)].index)

'''
def feature_processing(df):

    df['has_basement'] = df['sqft_basement'].apply(lambda x: 0 if x == 0 else 1)

    df['has_half_floor'] = df['floors'].apply(lambda x: 1 if x - math.floor(x) == 0.5 else 0)



    # yr_built => dayssincebuilt

    df['age'] = pd.to_datetime(df['date']).dt.year - df['yr_built']

    df['is_new'] = df['age'].apply(lambda x: 1 if x <= 0 else 0)



    room_sum = df.bedrooms + df.bathrooms + 1

    df['roomsize'] = df.sqft_living / room_sum

    df['bedroomsize'] = df.sqft_living / df.bedrooms

    

    df['total_rooms'] = df['bedrooms'] + df['bathrooms']

    df['roombybathroom'] = df['bedrooms'] / df['bathrooms']



    df['grade_condition'] = df['grade'] * df['condition']



    df['sqft_total'] = df['sqft_living'] + df['sqft_lot']

    df['sqft_total_size'] = df['sqft_living'] + df['sqft_lot'] + df['sqft_above'] + df['sqft_basement']

    df['sqft_total15'] = df['sqft_living15'] + df['sqft_lot15'] 

    df['sqft_total_size15'] = df['sqft_living15'] + df['sqft_lot15'] + df['sqft_above'] + df['sqft_basement']



    df['yr_renovated'] = df['yr_renovated'].apply(lambda x: np.nan if x == 0 else x)

    df['yr_renovated'] = df['yr_renovated'].fillna(df['yr_built'])

    df['is_renovated'] = df['yr_renovated'] - df['yr_built']

    df['is_renovated'] = df['is_renovated'].apply(lambda x: 0 if x <= 0 else 1)

    

    df['sqft_ratio'] = df['sqft_living'] / df['sqft_lot']

    df['sqft_ratio15'] = df['sqft_living15'] / df['sqft_lot15']     

    df['sqft_total_by_lot'] = (df['sqft_living'] + df['sqft_above'] + df['sqft_basement'])/df['sqft_lot']

    df['sqft_total_by_lot15'] = (df['sqft_living15'] + df['sqft_above'] + df['sqft_basement'])/df['sqft_lot15']

    

    qcut_count = 20

    df['qcut_long'] = pd.qcut(df['long'], qcut_count, labels=range(qcut_count))

    df['qcut_lat'] = pd.qcut(df['lat'], qcut_count, labels=range(qcut_count))

    df['qcut_long'] = df['qcut_long'].astype(int)

    df['qcut_lat'] = df['qcut_lat'].astype(int)



    df['date'] = pd.to_datetime(df['date'])

    df['yearmonth'] = df['date'].dt.year*100 + df['date'].dt.month

    df['month'] = df['date'].dt.month    

    df['date'] = df['date'].astype('int')

    df['date'] = df['date'].apply(str)

    

    df['date(new)'] = df['date'].apply(lambda x: int(x[4:8])+800 if x[:4] == '2015' else int(x[4:8])-400)

    df['how_old'] = df['date'].apply(lambda x: x[:4]).astype(int) - df[['yr_built', 'yr_renovated']].max(axis=1)

    del df['date']

    del df['yr_renovated']

    df['yr_built'] = df['yr_built'] - 1900

    df['sqft_floor'] = df['sqft_above'] / df['floors']

    

    return df
all_df = pd.concat([df_train_q, df_test_q])

all_df = feature_processing(all_df)
df_test_q = all_df.loc[all_df['price'].isnull()]

df_train_q = all_df.loc[all_df['price'].notnull()]
df_train_q['per_price'] = df_train_q['price'] / df_train_q['sqft_total_size']

zipcode_price = df_train_q.groupby(['zipcode'])['per_price'].agg({'mean','var'}).reset_index()

df_train_q = pd.merge(df_train_q, zipcode_price, how='left', on='zipcode')

df_test_q = pd.merge(df_test_q, zipcode_price, how='left', on='zipcode')



for df in [df_train_q, df_test_q]:

    df['mean'] = df['mean'] * df['sqft_total_size']

    df['var'] = df['var'] * df['sqft_total_size']

    

df_train_q = df_train_q.drop(columns=['per_price'])
group_df = groupby_helper(df_train_q, 'grade', 'price', ['mean'])

df_train_q = df_train_q.merge(group_df, on='grade', how='left')

df_test_q = df_test_q.merge(group_df, on='grade', how='left')



group_df = groupby_helper(df_train_q, 'bedrooms', 'price', ['mean'])

df_train_q = df_train_q.merge(group_df, on='bedrooms', how='left')

df_test_q = df_test_q.merge(group_df, on='bedrooms', how='left')



group_df = groupby_helper(df_train_q, 'bathrooms', 'price', ['mean'])

df_train_q = df_train_q.merge(group_df, on='bathrooms', how='left')

df_test_q = df_test_q.merge(group_df, on='bathrooms', how='left')
# 0 : 시애틀

# 1 : 레드먼드

# 2 : 벨뷰

# 3 : 이사콰

# 4 : 스노퀄미

# 5 : 스노퀄미 패스

# 6 : 렌턴

# 7 : 시택

# 8 : 켄트

# 9 : 페더럴웨이

# 10 : 버클리

cities = [[47.6127, -122.3333],

 [47.6697,-122.1997],

 [47.6133,-122.1944],

 [47.5317,-122.0339],

[47.5294,-121.8297],

[47.4083,-121.4014],

[47.4839,-122.2165],

[47.4427,-122.2883],

[47.3833,-122.2453],

[47.3251,-122.3098],

[47.1694,-122.0208]]



cities_lat = [i[0] for i in cities]

cities_long = [i[1] for i in cities]



print(cities_lat)

print(cities_long)

city_idx = list(range(11))

print(city_idx)

df_cities = pd.DataFrame({'cities_lat':cities_lat, 'cities_long':cities_long, 'city_idx':city_idx})



from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(df_cities[['cities_lat','cities_long']],df_cities['city_idx'])
pred_cities = knn.predict(df_train_q[['lat','long']])

df_train_q['city_idx'] = pred_cities

pred_cities = knn.predict(df_test_q[['lat','long']])

df_test_q['city_idx'] = pred_cities



df_test_q.plot(kind='scatter',x='long',y='lat', alpha = 0.2,

       marker='o', c='city_idx', cmap=plt.get_cmap('jet'),figsize=(15,8))
import scipy as sp



cor_abs = abs(df_train_q.corr(method='spearman')) 

cor_cols = cor_abs.nlargest(n=10, columns='price').index # price과 correlation이 높은 column 10개 뽑기(내림차순)

# spearman coefficient matrix

cor = np.array(sp.stats.spearmanr(df_train_q[cor_cols].values))[0] # 10 x 10

print(cor_cols.values)

plt.figure(figsize=(10,10))

sns.set(font_scale=1.25)

sns.heatmap(cor, fmt='.2f', annot=True, square=True , annot_kws={'size' : 8} ,xticklabels=cor_cols.values, yticklabels=cor_cols.values)
def rmse(y_true, y_pred):

    return np.sqrt(mean_squared_error(np.expm1(y_true), np.expm1(y_pred)))
for df in [df_train_q,df_test_q]:

    df['city_idx'] = df['city_idx'].apply(str)

    df['zipcode'] = df['zipcode'].apply(str)

    df['is_renovated'] = df['is_renovated'].apply(str)

    df['waterfront'] = df['waterfront'].apply(str)

    df['view'] = df['view'].apply(str)

    df['floors'] = df['floors'].apply(str)

    df['condition'] = df['condition'].apply(str)

    df['is_new'] = df['is_new'].apply(str)    

    df['has_basement'] = df['has_basement'].apply(str)    

    df['has_half_floor'] = df['has_half_floor'].apply(str)        

    df['month'] = df['month'].apply(str)        
x_train_q = df_train_q.copy()

x_train_q['price'] = np.log1p(x_train_q['price'])

x_train_q.loc[np.isinf(x_train_q['roombybathroom']),'roombybathroom'] = -1



x_test_q = df_test_q.copy()

x_test_q.loc[np.isinf(x_test_q['roombybathroom']),'roombybathroom'] = -1

area_feature = ['sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement', 'sqft_living15', 'sqft_lot15']

for col in area_feature:

    continous_feature_distribution(x_train_q, x_test_q, col)
x_train_q.loc[x_train_q['sqft_living'] > 13000]
x_train_q.loc[(x_train_q['price']>12) & (x_train_q['grade'] == 3)]
x_train_q.loc[(x_train_q['price']>14.7) & (x_train_q['grade'] == 8)]
x_train_q.loc[(x_train_q['price']>15.5) & (x_train_q['grade'] == 11)]
x_train_q.loc[x_train_q.roombybathroom.isna()]
y_train_org = x_train_q['price']
x_train_q = x_train_q.set_index('id')

x_test_q = x_test_q.set_index('id')

del x_train_q['price']

del x_train_q['bedrooms_price_mean']

del x_train_q['bathrooms_price_mean']

del x_test_q['price']

del x_test_q['bedrooms_price_mean']

del x_test_q['bathrooms_price_mean']
print('get_dummies() 수행 전 데이터 Shape:', x_train_q.shape)

x_train_ohe = pd.get_dummies(x_train_q)

print('get_dummies() 수행 후 데이터 Shape:', x_train_ohe.shape)



null_column_count = x_train_ohe.isnull().sum()[x_train_ohe.isnull().sum() > 0]

print('## Null 피처의 Type :\n', x_train_ohe.dtypes[null_column_count.index])

print('get_dummies() 수행 전 데이터 Shape:', x_test_q.shape)

x_test_ohe = pd.get_dummies(x_test_q)

print('get_dummies() 수행 후 데이터 Shape:', x_test_ohe.shape)



null_column_count = x_test_ohe.isnull().sum()[x_test_ohe.isnull().sum() > 0]

print('## Null 피처의 Type :\n', x_test_ohe.dtypes[null_column_count.index])

def get_rmse(model, X_test, y_test):

    pred = model.predict(X_test)

    mse = mean_squared_error(y_test, pred)

    rmse = np.sqrt(mse)

    print('{0} 로그 변환된 RMSE: {1}'.format(model.__class__.__name__,np.round(rmse, 3)))

    return rmse



def get_rmses(models, X_test, y_test):

    rmses = [ ]

    for model in models:

        rmse = get_rmse(model, X_test, y_test)

        rmses.append(rmse)

    return rmses
from sklearn.linear_model import LinearRegression, Ridge, Lasso

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error



y_target = y_train_org

X_features = x_train_ohe



X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.2, random_state=156)



# LinearRegression, Ridge, Lasso 학습, 예측, 평가

lr_reg = LinearRegression()

lr_reg.fit(X_train, y_train)



ridge_reg = Ridge()

ridge_reg.fit(X_train, y_train)



lasso_reg = Lasso()

lasso_reg.fit(X_train, y_train)



models = [lr_reg, ridge_reg, lasso_reg]

get_rmses(models, X_test, y_test)
def get_top_bottom_coef(model):

    # coef_ 속성을 기반으로 Series 객체를 생성. index는 컬럼명. 

    coef = pd.Series(model.coef_, index=X_features.columns)

    

    # + 상위 10개 , - 하위 10개 coefficient 추출하여 반환.

    coef_high = coef.sort_values(ascending=False).head(10)

    coef_low = coef.sort_values(ascending=False).tail(10)

    return coef_high, coef_low
def visualize_coefficient(models):

    # 3개 회귀 모델의 시각화를 위해 3개의 컬럼을 가지는 subplot 생성

    fig, axs = plt.subplots(figsize=(24,10),nrows=1, ncols=3)

    fig.tight_layout() 

    # 입력인자로 받은 list객체인 models에서 차례로 model을 추출하여 회귀 계수 시각화. 

    for i_num, model in enumerate(models):

        # 상위 10개, 하위 10개 회귀 계수를 구하고, 이를 판다스 concat으로 결합. 

        coef_high, coef_low = get_top_bottom_coef(model)

        coef_concat = pd.concat( [coef_high , coef_low] )

        # 순차적으로 ax subplot에 barchar로 표현. 한 화면에 표현하기 위해 tick label 위치와 font 크기 조정. 

        axs[i_num].set_title(model.__class__.__name__+' Coeffiecents', size=25)

        axs[i_num].tick_params(axis="y",direction="in", pad=-120)

        for label in (axs[i_num].get_xticklabels() + axs[i_num].get_yticklabels()):

            label.set_fontsize(22)

        sns.barplot(x=coef_concat.values, y=coef_concat.index , ax=axs[i_num])



# 앞 예제에서 학습한 lr_reg, ridge_reg, lasso_reg 모델의 회귀 계수 시각화.    

models = [lr_reg, ridge_reg, lasso_reg]

visualize_coefficient(models)
from sklearn.model_selection import cross_val_score



def get_avg_rmse_cv(models):

    for model in models:

        # 분할하지 않고 전체 데이터로 cross_val_score( ) 수행. 모델별 CV RMSE값과 평균 RMSE 출력

        rmse_list = np.sqrt(-cross_val_score(model, X_features, y_target,

                                             scoring="neg_mean_squared_error", cv = 5))

        rmse_avg = np.mean(rmse_list)

        print('\n{0} CV RMSE 값 리스트: {1}'.format( model.__class__.__name__, np.round(rmse_list, 3)))

        print('{0} CV 평균 RMSE 값: {1}'.format( model.__class__.__name__, np.round(rmse_avg, 3)))



# 앞 예제에서 학습한 lr_reg, ridge_reg, lasso_reg 모델의 CV RMSE값 출력           

models = [lr_reg, ridge_reg, lasso_reg]

get_avg_rmse_cv(models)
from sklearn.model_selection import GridSearchCV



def get_best_params(model, params):

    grid_model = GridSearchCV(model, param_grid=params, 

                              scoring='neg_mean_squared_error', cv=5)

    grid_model.fit(X_features, y_target)

    rmse = np.sqrt(-1* grid_model.best_score_)

    print('{0} 5 CV 시 최적 평균 RMSE 값: {1}, 최적 alpha:{2}'.format(model.__class__.__name__,

                                        np.round(rmse, 4), grid_model.best_params_))

    return grid_model.best_estimator_



ridge_params = { 'alpha':[0.05, 0.1, 1, 5, 8, 10, 12, 15, 20] }

lasso_params = { 'alpha':[0.001, 0.005, 0.008, 0.05, 0.03, 0.1, 0.5, 1,5, 10] }

best_rige = get_best_params(ridge_reg, ridge_params)

best_lasso = get_best_params(lasso_reg, lasso_params)

# 앞의 최적화 alpha값으로 학습데이터로 학습, 테스트 데이터로 예측 및 평가 수행. 

lr_reg = LinearRegression()

lr_reg.fit(X_train, y_train)

ridge_reg = Ridge(alpha=0.1)

ridge_reg.fit(X_train, y_train)

lasso_reg = Lasso(alpha=0.001)

lasso_reg.fit(X_train, y_train)



# 모든 모델의 RMSE 출력

models = [lr_reg, ridge_reg, lasso_reg]

get_rmses(models, X_test, y_test)



# 모든 모델의 회귀 계수 시각화 

models = [lr_reg, ridge_reg, lasso_reg]

visualize_coefficient(models)
from scipy.stats import skew



# object가 아닌 숫자형 피쳐의 컬럼 index 객체 추출.

features_index = x_train_q.dtypes[x_test_q.dtypes != 'object'].index

# house_df에 컬럼 index를 [ ]로 입력하면 해당하는 컬럼 데이터 셋 반환. apply lambda로 skew( )호출 

skew_features = x_train_q[features_index].apply(lambda x : skew(x))

# skew 정도가 5 이상인 컬럼들만 추출. 

skew_features_top = skew_features[skew_features > 1]



# waterfront, is_renovated, view, condition

skew_features_top = skew_features_top[~skew_features_top.index.isin([

    'waterfront', 'is_renovated', 'view', 'condition', 'per_price', 'price', 'is_new', 'bathrooms_25', 'bathrooms_75', 'bedroomsize', 'has_half_floor'])]

print(skew_features_top.sort_values(ascending=False))



x_train_sq = x_train_q.copy()

x_test_sq = x_test_q.copy()



x_train_sq[skew_features_top.index] = np.log1p(x_train_sq[skew_features_top.index])

x_test_sq[skew_features_top.index] = np.log1p(x_test_sq[skew_features_top.index])



x_train_ohe = pd.get_dummies(x_train_sq)

x_test_ohe = pd.get_dummies(x_test_sq)

print('get_dummies() 수행 전 데이터 Shape:', x_train_sq.shape)

x_train_ohe = pd.get_dummies(x_train_sq)

print('get_dummies() 수행 후 데이터 Shape:', x_train_ohe.shape)



null_column_count = x_train_ohe.isnull().sum()[x_train_ohe.isnull().sum() > 0]

print('## Null 피처의 Type :\n', x_train_ohe.dtypes[null_column_count.index])

print('get_dummies() 수행 전 데이터 Shape:', x_test_sq.shape)

x_test_ohe = pd.get_dummies(x_test_sq)

print('get_dummies() 수행 후 데이터 Shape:', x_test_ohe.shape)



null_column_count = x_test_ohe.isnull().sum()[x_test_ohe.isnull().sum() > 0]

print('## Null 피처의 Type :\n', x_test_ohe.dtypes[null_column_count.index])

# Skew가 높은 피처들을 로그 변환 했으므로 다시 원-핫 인코딩 적용 및 피처/타겟 데이터 셋 생성,

y_target = y_train_org

X_features = x_train_sq



X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.2, random_state=156)



# test set 을 적용

'''

X_train = X_features

y_train = y_target

X_test = test_df

'''



from sklearn.model_selection import GridSearchCV



def get_best_params(model, params):

    grid_model = GridSearchCV(model, param_grid=params, 

                              scoring='neg_mean_squared_error', cv=5)

    grid_model.fit(X_features, y_target)

    rmse = np.sqrt(-1* grid_model.best_score_)

    print('{0} 5 CV 시 최적 평균 RMSE 값: {1}, 최적 alpha:{2}'.format(model.__class__.__name__,

                                        np.round(rmse, 4), grid_model.best_params_))

    return grid_model.best_estimator_



ridge_params = { 'alpha':[0.05, 0.1, 1, 5, 8, 10, 12, 15, 20] }

lasso_params = { 'alpha':[0.001, 0.005, 0.008, 0.05, 0.03, 0.1, 0.5, 1,5, 10] }

best_ridge = get_best_params(ridge_reg, ridge_params)

best_lasso = get_best_params(lasso_reg, lasso_params)
np.where(y_target.values >= np.finfo(np.float64).max)
fig, ax = plt.subplots(5,4,figsize=(20,20))

cols = X_train.columns

n=0

for r in range(5):

    for c in range(4):

        sns.kdeplot(X_train[cols[n]], ax=ax[r][c])

        ax[r][c].set_title(cols[n], fontsize=20)

        n+=1

        if n==X_train.shape[1]:

            break

# 앞의 최적화 alpha값으로 학습데이터로 학습, 테스트 데이터로 예측 및 평가 수행. 

lr_reg = LinearRegression()

lr_reg.fit(X_train, y_train)

ridge_reg = Ridge(alpha=0.1)

ridge_reg.fit(X_train, y_train)

lasso_reg = Lasso(alpha=0.001)

lasso_reg.fit(X_train, y_train)



# 모든 모델의 RMSE 출력

models = [lr_reg, ridge_reg, lasso_reg]

get_rmses(models, X_test, y_test)



# 모든 모델의 회귀 계수 시각화 

models = [lr_reg, ridge_reg, lasso_reg]

visualize_coefficient(models)
plt.scatter(x = x_train_ohe['sqft_total_size'], y = y_train_org)

plt.ylabel('SalePrice', fontsize=15)

plt.xlabel('sqft_above', fontsize=15)

plt.show()

# 앞의 최적화 alpha값으로 학습데이터로 학습, 테스트 데이터로 예측 및 평가 수행. 

lr_reg = LinearRegression()

lr_reg.fit(X_train, y_train)

ridge_reg = Ridge(alpha=0.1)

ridge_reg.fit(X_train, y_train)

lasso_reg = Lasso(alpha=0.001)

lasso_reg.fit(X_train, y_train)



# 모든 모델의 RMSE 출력

models = [lr_reg, ridge_reg, lasso_reg]

get_rmses(models, X_test, y_test)



# 모든 모델의 회귀 계수 시각화 

models = [lr_reg, ridge_reg, lasso_reg]

visualize_coefficient(models)
import xgboost as xgb



xgb_params = {

    'eta': 0.01,

    'max_depth': 6,

    'subsample': 0.8,

    'colsample_bytree': 0.8,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'silent': 1

}



y_target = y_train_org.copy()

X_features = x_train_ohe



print('Transform DMatrix...')

dtrain = xgb.DMatrix(X_features, y_target)

dtest = xgb.DMatrix(x_test_ohe)



print('Start Cross Validation...')



cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=5000, early_stopping_rounds=20, verbose_eval=50, show_stdv=False)

cv_output[['train-rmse-mean', 'test-rmse-mean']].plot()

print('best num_boost_rounds = ', len(cv_output))

rounds = len(cv_output)
model = xgb.train(xgb_params, dtrain, num_boost_round=rounds)

y_pred = model.predict(dtest)
dfeatures = xgb.DMatrix(X_features)

target_pred = model.predict(dfeatures)

import numpy as np

predict_price = np.expm1(y_pred)

predict_price



x_test_ohe['price'] = predict_price

x_test_ohe[['price']].to_csv('jacob_predicted_xg_20190505_3.csv')