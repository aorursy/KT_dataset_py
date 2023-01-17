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
from collections import Counter

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression, ElasticNet, Lasso,  BayesianRidge, LassoLarsIC

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, RobustScaler

from sklearn.pipeline import make_pipeline

from sklearn.tree import DecisionTreeRegressor

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

import lightgbm as lgb

#import xgboost as xgb



from datetime import tzinfo, timedelta, datetime

import seaborn as sns

from matplotlib import pyplot as plt

%matplotlib inline



from scipy import stats

from scipy.stats import norm, skew #for some statistics



from sklearn import model_selection, preprocessing, metrics
#load all files

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

df_hist = pd.read_csv('../input/historical_transactions.csv')

df_new = pd.read_csv('../input/new_merchant_transactions.csv')

df_merch = pd.read_csv('../input/merchants.csv')
#drop duplicated from the merchent list

df_merch = df_merch.drop_duplicates(['merchant_id'],keep='first')

df_merch.columns
# I am interested in 2 featres from the merchant data

df_merch = df_merch[['merchant_id','numerical_1', 'numerical_2']]

#Combine the data with the card details

df_hist = pd.merge(df_hist,df_merch, on='merchant_id',how='left',sort=False)

df_new = pd.merge(df_new,df_merch, on='merchant_id',how='left',sort=False)
df_merch = pd.DataFrame()
#save the card details for future use

dy_train = df_train['target']

df_train = df_train.drop(['target'],axis=1)

test_card_id = df_test['card_id']
df_merge = pd.concat([df_train,df_test],sort=False)
hist_df = pd.concat([df_hist,df_new],sort=False)
df_hist = hist_df
#def aggregate - authorized_flag = Y

xin = df_hist[df_hist['authorized_flag'] == 'Y'].groupby(['card_id'])



xon = xin.agg({'purchase_amount':['min','max','mean','sum']})

xon.columns = ["_y_".join(x) for x in xon.columns.ravel()]

df_merge = pd.merge(df_merge,xon, on='card_id',how='left',sort=False)



xon = xin.agg({'authorized_flag':['count']})

xon.columns = ["_y_".join(x) for x in xon.columns.ravel()]

df_merge = pd.merge(df_merge,xon, on='card_id',how='left',sort=False)





xon = xin.agg({'numerical_1':['sum']})

xon.columns = ["_y_".join(x) for x in xon.columns.ravel()]

df_merge = pd.merge(df_merge,xon, on='card_id',how='left',sort=False)



xon = xin.agg({'numerical_2':['sum']})

xon.columns = ["_y_".join(x) for x in xon.columns.ravel()]

df_merge = pd.merge(df_merge,xon, on='card_id',how='left',sort=False)





#def aggregate - authorized_flag != Y

xin = df_hist[df_hist['authorized_flag'] != 'Y'].groupby(['card_id'])

xon = xin.agg({'authorized_flag':['count']})

xon.columns = ["_n_".join(x) for x in xon.columns.ravel()]

df_merge = pd.merge(df_merge,xon, on='card_id',how='left',sort=False)



xon = xin.agg({'purchase_amount':['min','max','mean','sum']})

xon.columns = ["_n_".join(x) for x in xon.columns.ravel()]

df_merge = pd.merge(df_merge,xon, on='card_id',how='left',sort=False)



xon = xin.agg({'numerical_1':['sum']})

xon.columns = ["_n_".join(x) for x in xon.columns.ravel()]

df_merge = pd.merge(df_merge,xon, on='card_id',how='left',sort=False)



xon = xin.agg({'numerical_2':['sum']})

xon.columns = ["_n_".join(x) for x in xon.columns.ravel()]

df_merge = pd.merge(df_merge,xon, on='card_id',how='left',sort=False)





xon = df_hist[df_hist['category_1'] == 'Y'].groupby(['card_id']).agg({'category_1':['count']})

xon.columns = ["_y_".join(x) for x in xon.columns.ravel()]

df_merge = pd.merge(df_merge,xon, on='card_id',how='left',sort=False)



xon = df_hist[df_hist['category_1'] != 'Y'].groupby(['card_id']).agg({'category_1':['count']})

xon.columns = ["_n_".join(x) for x in xon.columns.ravel()]

df_merge = pd.merge(df_merge,xon, on='card_id',how='left',sort=False)



xon = df_hist[df_hist['category_2'] == 1].groupby(['card_id']).agg({'category_2':['count']})

xon.columns = ["_1_".join(x) for x in xon.columns.ravel()]

df_merge = pd.merge(df_merge,xon, on='card_id',how='left',sort=False)



xon = df_hist[df_hist['category_2'] == 2].groupby(['card_id']).agg({'category_2':['count']})

xon.columns = ["_2_".join(x) for x in xon.columns.ravel()]

df_merge = pd.merge(df_merge,xon, on='card_id',how='left',sort=False)



xon = df_hist[df_hist['category_2'] == 3].groupby(['card_id']).agg({'category_2':['count']})

xon.columns = ["_3_".join(x) for x in xon.columns.ravel()]

df_merge = pd.merge(df_merge,xon, on='card_id',how='left',sort=False)



xon = df_hist[df_hist['category_2'] == 4].groupby(['card_id']).agg({'category_2':['count']})

xon.columns = ["_4_".join(x) for x in xon.columns.ravel()]

df_merge = pd.merge(df_merge,xon, on='card_id',how='left',sort=False)



xon = df_hist[df_hist['category_2'] == 5].groupby(['card_id']).agg({'category_2':['count']})

xon.columns = ["_5_".join(x) for x in xon.columns.ravel()]

df_merge = pd.merge(df_merge,xon, on='card_id',how='left',sort=False)



xon = df_hist[df_hist['category_3'] == 'A'].groupby(['card_id']).agg({'category_3':['count']})

xon.columns = ["_A_".join(x) for x in xon.columns.ravel()]

df_merge = pd.merge(df_merge,xon, on='card_id',how='left',sort=False)



xon = df_hist[df_hist['category_3'] == 'B'].groupby(['card_id']).agg({'category_3':['count']})

xon.columns = ["_B_".join(x) for x in xon.columns.ravel()]

df_merge = pd.merge(df_merge,xon, on='card_id',how='left',sort=False)



xon = df_hist[df_hist['category_3'] == 'C'].groupby(['card_id']).agg({'category_3':['count']})

xon.columns = ["_C_".join(x) for x in xon.columns.ravel()]

df_merge = pd.merge(df_merge,xon, on='card_id',how='left',sort=False)



xin = df_hist.groupby(['card_id'])

xon = xin.agg({'merchant_id':['nunique']})

xon.columns = ["_".join(x) for x in xon.columns.ravel()]

df_merge = pd.merge(df_merge,xon, on='card_id',how='left',sort=False)



xon = xin.agg({'subsector_id':['nunique']})

xon.columns = ["_".join(x) for x in xon.columns.ravel()]

df_merge = pd.merge(df_merge,xon, on='card_id',how='left',sort=False)



xon = xin.agg({'state_id':['nunique']})

xon.columns = ["_".join(x) for x in xon.columns.ravel()]

df_merge = pd.merge(df_merge,xon, on='card_id',how='left',sort=False)



xon = xin.agg({'installments':['sum']})

xon.columns = ["_".join(x) for x in xon.columns.ravel()]

df_merge = pd.merge(df_merge,xon, on='card_id',how='left',sort=False)



xon = xin.agg({'purchase_amount':['min','max','sum','mean']})

xon.columns = ["_".join(x) for x in xon.columns.ravel()]

df_merge = pd.merge(df_merge,xon, on='card_id',how='left',sort=False)



xon = xin.agg({'numerical_1':['min','max','sum','mean']}) 

xon.columns = ["_".join(x) for x in xon.columns.ravel()]

df_merge = pd.merge(df_merge,xon, on='card_id',how='left',sort=False)



xon = xin.agg({'numerical_2':['min','max','sum','mean']}) 

xon.columns = ["_".join(x) for x in xon.columns.ravel()]

df_merge = pd.merge(df_merge,xon, on='card_id',how='left',sort=False)



xon = xin.agg({'purchase_date':['min','max']}) 

xon.columns = ["_".join(x) for x in xon.columns.ravel()]

df_merge = pd.merge(df_merge,xon, on='card_id',how='left',sort=False)

#Saving for future need - instead of processing all again

#df_merge.to_csv('Merged_all.csv',index=False)
df_merge.isnull().sum()
df_merge['first_active_month'] = df_merge['first_active_month'].fillna(df_merge['first_active_month'].mode()[0])

df_merge.isnull().sum()
df_merge = df_merge.fillna(0)
#df_merge['TotalFeatures'] = df_merge['feature_1'] + df_merge['feature_2'] + df_merge['feature_3']

df_merge['auth'] = df_merge['authorized_flag_y_count'] + df_merge['authorized_flag_n_count']

df_merge['AuthRateY'] = (df_merge['purchase_amount_y_sum'] /df_merge['auth'])

df_merge['AuthRateN'] = (df_merge['purchase_amount_n_sum'] /df_merge['auth'])

df_merge['AvgAuthPurY'] = df_merge.apply(lambda row: row['purchase_amount_sum']/row['authorized_flag_y_count'] if(row['authorized_flag_y_count']>0) else 0, axis=1)

df_merge['AvgAuthPurN'] = df_merge.apply(lambda row: row['purchase_amount_sum']/row['authorized_flag_n_count'] if(row['authorized_flag_n_count']>0) else 0, axis=1)



df_merge['cat1'] = df_merge['category_1_y_count'] + df_merge['category_1_n_count']

df_merge['RatioCat1'] = df_merge['category_1_y_count'] / df_merge['cat1']

df_merge['cat2'] = df_merge['category_2_1_count'] + df_merge['category_2_2_count'] + df_merge['category_2_3_count'] + df_merge['category_2_4_count'] + df_merge['category_2_5_count']

df_merge['AvgCat2'] = df_merge['cat2'] / 5 

df_merge['cat3'] = df_merge['category_3_A_count'] + df_merge['category_3_B_count'] + df_merge['category_3_C_count']

df_merge['AvgCat3'] = df_merge['cat3'] / 3



df_merge['AvgInstAmt'] =  df_merge.apply(lambda row: row['purchase_amount_sum']/row['installments_sum'] if(row['installments_sum']>0) else row['purchase_amount_sum'], axis=1)

df_merge['AvgPurchAmt'] =  df_merge.apply(lambda row: row['purchase_amount_sum']/row['auth'], axis=1)

df_merge['AvgMerchant'] =  df_merge.apply(lambda row: row['purchase_amount_sum']/row['merchant_id_nunique'], axis=1)

df_merge['AvgSubsector'] =  df_merge.apply(lambda row: row['purchase_amount_sum']/row['subsector_id_nunique'], axis=1)
#find the number of month over

df_merge['YearElapsed'] = df_merge['first_active_month'].apply(lambda x: (2019- int(str(x).split('-')[0])))

df_merge['MonthElapsed'] = df_merge['first_active_month'].apply(lambda x: (12-(int(str(x).split('-')[1]))))

#find the number of month over

df_merge['YearActive'] = df_merge['first_active_month'].apply(lambda x: (int(str(x).split('-')[0])))

df_merge['MonthActive'] = df_merge['first_active_month'].apply(lambda x: ((int(str(x).split('-')[1]))))
df_merge.head()
df_merge['purchase_date_min'] = pd.to_datetime(df_merge['purchase_date_min'])

df_merge['purchase_date_max'] = pd.to_datetime(df_merge['purchase_date_max'])
#'purchase_date'

df_merge['days_from_purchase_first'] = (datetime.today() - df_merge['purchase_date_min']).dt.days

df_merge['days_from_purchase_last'] = (datetime.today() - df_merge['purchase_date_max']).dt.days



df_merge['first_purchase_year'] = df_merge['purchase_date_min'].dt.year

df_merge['first_purchase_month'] = df_merge['purchase_date_min'].dt.month

df_merge['first_purchase_year'] = df_merge['purchase_date_min'].dt.day

df_merge['first_purchase_wkofyear'] = df_merge['purchase_date_min'].dt.weekofyear

df_merge['first_purchase_hour'] = df_merge['purchase_date_min'].dt.hour



df_merge['last_purchase_year'] = df_merge['purchase_date_max'].dt.year

df_merge['last_purchase_month'] = df_merge['purchase_date_max'].dt.month

df_merge['last_purchase_year'] = df_merge['purchase_date_max'].dt.day

df_merge['last_purchase_wkofyear'] = df_merge['purchase_date_max'].dt.weekofyear

df_merge['last_purchase_hour'] = df_merge['purchase_date_max'].dt.hour
df_train = df_merge.iloc[:len(df_train), :]
#Find the mean of the train data to the whole of the merge data

df_merge['purchase_amount_y_meanDiff'] = df_merge['purchase_amount_y_mean']- df_train['purchase_amount_y_mean'].mean()

df_merge['purchase_amount_y_sumDiff'] = df_merge['purchase_amount_y_sum']- df_train['purchase_amount_y_sum'].mean()

df_merge['purchase_amount_y_minDiff'] = df_merge['purchase_amount_y_min']- df_train['purchase_amount_y_min'].mean()

df_merge['purchase_amount_y_maxDiff'] = df_merge['purchase_amount_y_max']- df_train['purchase_amount_y_max'].mean()



df_merge['purchase_amount_minDiff'] = df_merge['purchase_amount_min'] - df_train['purchase_amount_min'].mean()

df_merge['purchase_amount_maxDiff'] = df_merge['purchase_amount_max'] - df_train['purchase_amount_max'].mean()

df_merge['purchase_amount_sumDiff'] = df_merge['purchase_amount_sum'] - df_train['purchase_amount_sum'].mean()

df_merge['purchase_amount_meanDiff'] = df_merge['purchase_amount_mean'] - df_train['purchase_amount_mean'].mean()

#Purchase amount to 'purchase_amount_sum' 'merchant_id_nunique', 'subsector_id_nunique', 'state_id_nunique'

df_merge['merchant_purchase_mean'] = df_merge['purchase_amount_sum'] / df_merge['merchant_id_nunique']

df_merge['subsector_purchase_mean'] = df_merge['purchase_amount_sum'] / df_merge['subsector_id_nunique']

df_merge['state_purchase_mean'] = df_merge['purchase_amount_sum'] / df_merge['state_id_nunique']

df_merge['anual_purchase_mean'] = df_merge['purchase_amount_sum'] / df_merge['YearElapsed']

df_merge['month_purchase_mean'] = df_merge['purchase_amount_sum'] / (df_merge['YearElapsed']*12 + df_merge['MonthElapsed'])



#Purchase amout and numerical 1 & 2 value

df_merge['purchase_numerical1_rate'] = df_merge['purchase_amount_sum'] / df_merge['numerical_1_sum']

df_merge['purchase_numerical2_rate'] = df_merge['purchase_amount_sum'] / df_merge['numerical_2_sum']

droplist = ['purchase_date_min','purchase_date_max']#,'auth','auth_n','AuthRateY', 'AuthRateN', 'cat1', 'RatioCat1', 'cat2', 'AvgCat2',

#       'cat3', 'AvgCat3', 'AvgPurchAmt','merchant_count']

df_merge = df_merge.drop(droplist,axis=1)
numeric_feats = df_merge.dtypes[df_merge.dtypes != "object"].index

# Check the skew of all numerical features

skewed_feats = df_merge[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

print("\nSkew in numerical features: \n")

skewness = pd.DataFrame({'Skew' :skewed_feats})

skewness = skewness[abs(skewness.Skew) > 0.8]

skewness.head(25)
skewness = skewness[abs(skewness.Skew) > 1.0]

print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))



from scipy.special import boxcox1p

skewed_features = skewness.index

lam = 0.15

for feat in skewed_features:

    #all_data[feat] += 1

    df_merge[feat] = boxcox1p(df_merge[feat], lam)
df_merge.isnull().sum()[df_merge.isnull().sum() > 0]
#Fill the null values after the boxcoxing 

def fillNa(lst,val):

    for x in list(lst):

        df_merge[x] = df_merge[x].fillna(0)

    

        

lst =['numerical_1_y_sum','numerical_2_y_sum','purchase_amount_n_sum','numerical_1_n_sum','numerical_2_n_sum','numerical_1_sum',

      'numerical_2_sum','installments_sum','AvgAuthPurY','AvgAuthPurN','AvgInstAmt','AvgMerchant','AvgSubsector','purchase_amount_y_maxDiff',

     'purchase_amount_sumDiff','purchase_amount_meanDiff','purchase_amount_maxDiff','purchase_amount_sum','merchant_purchase_mean',

     'subsector_purchase_mean','state_purchase_mean','anual_purchase_mean','month_purchase_mean','purchase_numerical1_rate','purchase_numerical2_rate']

        

fillNa(lst,0)

#Drop the features having high amout of null values

to_drop = ['purchase_amount_n_sum','purchase_amount_sum',

      'AvgAuthPurN','AvgInstAmt','AvgMerchant','AvgSubsector','purchase_amount_y_maxDiff','purchase_amount_maxDiff',

      'purchase_amount_sumDiff','purchase_amount_meanDiff','merchant_purchase_mean','subsector_purchase_mean',

      'state_purchase_mean','anual_purchase_mean','month_purchase_mean','purchase_numerical1_rate','purchase_numerical2_rate']

df_merge = df_merge.drop(to_drop,axis=1)
df_merge.columns
#convert float to numbers whereever not float

feat = ['authorized_flag_y_count','authorized_flag_n_count', 'category_1_y_count', 'category_1_n_count', 

        'category_2_1_count','category_2_2_count', 'category_2_3_count', 'category_2_4_count',

       'category_2_5_count', 'category_3_A_count', 'category_3_B_count','category_3_C_count', 

        'merchant_id_nunique', 'subsector_id_nunique','state_id_nunique','YearElapsed', 'MonthElapsed']

for z in feat:

    if df_merge[z].dtype =='float':

        df_merge[z] = df_merge[z].astype(int)

        

feat_cat = ['feature_1', 'feature_2', 'feature_3']

for z in feat_cat:

    df_merge[z] = df_merge[z].astype('category')
df_cat = df_merge.select_dtypes(include=['category'])

df_num = df_merge.select_dtypes(exclude=['category'])

#apply label encoder for categorical columns 

le = LabelEncoder()

df_cat = df_cat.apply(le.fit_transform)

#perform one hot encoding on categorical columns 

#df_cat_dum = pd.get_dummies(df_cat.astype(str))

#Concatenate dummy columns, numeric and the target (dependent variable) as a final dataframe

df_merge = pd.concat([df_cat,df_num],axis=1)
def score_model(model):

    model.fit(Xtrain,ytrain)

    ypred = model.predict(Xtest)

    return mean_squared_error(ytest,ypred)



def pred_model(model):

    model.fit(df_train,dy_train)

    ypred = model.predict(df_test)

    return ypred
droplist = ['first_active_month','card_id']#,'auth','auth_n','AuthRateY', 'AuthRateN', 'cat1', 'RatioCat1', 'cat2', 'AvgCat2',

#       'cat3', 'AvgCat3', 'AvgPurchAmt','merchant_count']

df_merged = df_merge.drop(droplist,axis=1)

df_train = df_merged.iloc[:len(df_train), :]

df_test  = df_merged.iloc[len(df_train):, :]
Xtrain,Xtest,ytrain,ytest = train_test_split(df_train,dy_train,test_size=0.2,random_state=42)
LinReg = LinearRegression(n_jobs=6)

np.sqrt(score_model(LinReg))
BasRig = BayesianRidge()

np.sqrt(score_model(BasRig))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.5, random_state=3))

np.sqrt(score_model(ENet))
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))

np.sqrt(score_model(lasso))
df_train.columns
cols_to_use =['feature_1', 'feature_2', 'feature_3', 'purchase_amount_y_min',

       'purchase_amount_y_max', 'purchase_amount_y_mean',

       'purchase_amount_y_sum', 'authorized_flag_y_count', 'numerical_1_y_sum',

       'numerical_2_y_sum', 'authorized_flag_n_count', 'purchase_amount_n_min',

       'purchase_amount_n_max', 'purchase_amount_n_mean', 'numerical_1_n_sum',

       'numerical_2_n_sum', 'category_1_y_count', 'category_1_n_count',

       'category_2_1_count', 'category_2_2_count', 'category_2_3_count',

       'category_2_4_count', 'category_2_5_count', 'category_3_A_count',

       'category_3_B_count', 'category_3_C_count', 'merchant_id_nunique',

       'subsector_id_nunique', 'state_id_nunique', 'installments_sum',

       'purchase_amount_min', 'purchase_amount_max', 'purchase_amount_mean',

       'numerical_1_min', 'numerical_1_max', 'numerical_1_sum',

       'numerical_1_mean', 'numerical_2_min', 'numerical_2_max',

       'numerical_2_sum', 'numerical_2_mean', 'auth', 'AuthRateY', 'AuthRateN',

       'AvgAuthPurY', 'cat1', 'RatioCat1', 'cat2', 'AvgCat2', 'cat3',

       'AvgCat3', 'AvgPurchAmt', 'YearElapsed', 'MonthElapsed', 'YearActive',

       'MonthActive', 'days_from_purchase_first', 'days_from_purchase_last',

       'first_purchase_year', 'first_purchase_month',

       'first_purchase_wkofyear', 'first_purchase_hour', 'last_purchase_year',

       'last_purchase_month', 'last_purchase_wkofyear', 'last_purchase_hour',

       'purchase_amount_y_meanDiff', 'purchase_amount_y_sumDiff',

       'purchase_amount_y_minDiff', 'purchase_amount_minDiff']



def run_lgb(train_X, train_y, val_X, val_y, test_X):

    params = {'num_leaves': 31,

         'min_data_in_leaf': 30, 

         'objective':'regression',

         'max_depth': -1,

         'learning_rate': 0.01,

         "min_child_samples": 20,

         "boosting": "gbdt",

         "feature_fraction": 0.9,

         "bagging_freq": 1,

         "bagging_fraction": 0.9 ,

         "bagging_seed": 11,

         "metric": 'rmse',

         "lambda_l1": 0.1,

         "verbosity": -1,

         "nthread": 4,

         "random_state": 2018}

    

    lgtrain = lgb.Dataset(train_X, label=train_y)

    lgval = lgb.Dataset(val_X, label=val_y)

    evals_result = {}

    model = lgb.train(params, lgtrain, 1000, valid_sets=[lgval], early_stopping_rounds=100, verbose_eval=100, evals_result=evals_result)

    

    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)

    return pred_test_y, model, evals_result



train_X = df_train[cols_to_use]

test_X = df_test[cols_to_use]

train_y = dy_train.values



pred_test = 0

modval = 4.0

kf = model_selection.KFold(n_splits=5, random_state=2018, shuffle=True)

for dev_index, val_index in kf.split(df_train):

    dev_X, val_X = train_X.loc[dev_index,:], train_X.loc[val_index,:]

    dev_y, val_y = train_y[dev_index], train_y[val_index]

    

    pred_test_tmp, model, evals_result = run_lgb(dev_X, dev_y, val_X, val_y, test_X)

    m = float(str(model.best_score).split(':')[2].split('}')[0])

    print(m)

    if (m < modval):

        pred_test = pred_test_tmp

        modval = m

    

#pred_test /= 5.

#np.sqrt(score_model(GBoost))

print("model value :",modval)
pred_df = pd.DataFrame({'card_id':test_card_id})

pred_df['target'] = pred_test #pred_model(BasRig)
pred_df.to_csv('prediction001.csv',index=False)
pred_df.head()