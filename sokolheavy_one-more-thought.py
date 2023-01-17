import warnings

warnings.filterwarnings("ignore")



import pandas as pd

import numpy as np

from numpy import percentile

from numpy import nanmedian



import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

from scipy.stats import skew

from scipy.stats import mode

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, RobustScaler, StandardScaler

from sklearn.model_selection import cross_val_score, cross_val_predict

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics.scorer import make_scorer

from sklearn.model_selection import KFold



from xgboost import XGBRegressor

import lightgbm as lgb

from lightgbm import LGBMRegressor

from catboost import Pool, CatBoostRegressor



## Hyperopt modules

import gc

from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK, STATUS_RUNNING

from functools import partial



from sklearn.externals import joblib 

pd.set_option('display.max_columns', 500)

pd.set_option('display.max_rows', 500)
df = pd.read_csv('../input/TTiDS20/train.csv').drop(columns=['Unnamed: 0'])

test = pd.read_csv('../input/TTiDS20/test_no_target.csv').drop(columns=['Unnamed: 0'])

zipcodes = pd.read_csv('../input/TTiDS20/zipcodes.csv').drop(columns=['Unnamed: 0'])

sample_submission = pd.read_csv('../input/TTiDS20/sample_submission.csv')
#skew<1.88 - norm

print(df.price.skew())



plt.figure(figsize=(8,15))



plt.subplot(3,1,1)

plt.title('Car Price Distribution Plot')

sns.distplot(df.price)



plt.subplot(3,1,2)

plt.title('Car Price Spread')

sns.boxplot(y=df.price)



plt.subplot(3,1,3)

_ = stats.probplot(df['price'], plot=plt)

plt.title("Probability plot: SalePrice")

plt.show()

df['price'] = np.log1p(df['price'])
print(df.price.skew())



plt.figure(figsize=(8,15))



plt.subplot(3,1,1)

plt.title('Car Price Distribution Plot')

sns.distplot(np.log1p(df.price))



plt.subplot(3,1,2)

plt.title('Car Price Spread')

sns.boxplot(np.log1p(df.price))



plt.subplot(3,1,3)

_ = stats.probplot(df['price'], plot=plt)

plt.title("Probability plot: SalePrice")

plt.show()
df_zip = zipcodes.groupby('zipcode').agg({'city': ['count'],

                                          'latitude': ['mean'],

                                          'longitude': ['mean']})



df_zip.columns = df_zip.columns.map('_'.join)

df_zip = df_zip.fillna(0).reset_index()

df_zip['zip_size'] = np.where(df_zip.city_count>20, 'L', np.where(df_zip.city_count>5, 'M', 'S'))



df = df.merge(df_zip, on ='zipcode', how = 'left')

col_na = ['city_count', 'latitude_mean', 'longitude_mean']

for i in col_na:

    df[i].fillna(df[i].mean(), inplace = True)

    

    

test = test.merge(df_zip, on ='zipcode', how = 'left')

for i in col_na:

    test[i].fillna(test[i].mean(), inplace = True)    



df.drop('zipcode', axis = 1, inplace = True)

test.drop('zipcode', axis = 1, inplace = True)
def reg_year_transform(my_df, reg_col):

    my_df[reg_col] = my_df[reg_col].apply(lambda x: '19' + str(x) if len(str(x))==2 else 

                                                    '200' + str(x) if len(str(x) ) == 1 else x)

    my_df[reg_col] = my_df[reg_col].astype('int')

    my_df['IsOld'] = np.where(my_df[reg_col] < 2000, 1, 0)

    my_df['YearCar'] = my_df[reg_col].max() - my_df[reg_col]

    my_df['reg_year_str'] = my_df[reg_col].astype('str')

    return my_df

    

df = reg_year_transform(df, 'registration_year')  

test = reg_year_transform(test, 'registration_year') 
def fill_missings(my_df):    

    my_df['zip_size'] = my_df['zip_size'].fillna(my_df['zip_size'].mode()[0])

    my_df['gearbox'] = my_df['gearbox'].fillna(my_df['gearbox'].mode()[0])

    my_df['type'] = my_df['type'].fillna('other')

    my_df['fuel'] = my_df['fuel'].fillna(my_df['fuel'].mode()[0])

    my_df['damage'] = my_df['damage'].fillna(0.0)

    my_df['model'] = my_df['model'].fillna(my_df.groupby(['brand'])['model']\

                          .transform(lambda x: 'other' if mode(x)[0][0] == 0 else mode(x)[0][0]))

    return my_df

    

df = fill_missings(df)

test = fill_missings(test) 
%%time

cats = ['insurance_price', 'engine_capacity']

for cat in cats:

    print("Category: {}".format(cat))

    tmp = df.groupby(['model', 'type', 'registration_year']).agg({cat: ['mean', 'median']}) 

    tmp.columns = tmp.columns.map('_'.join)

    

    df = df.merge(tmp, how='left', on=['model', 'type', 'registration_year'])

    df[cat + '_mean'].fillna(np.nanmean(df[cat + '_mean']), inplace = True)

    df[cat + '_median'].fillna(np.nanmedian(df[cat + '_median']), inplace = True)

    df[cat].fillna(np.nanmedian(df[cat + '_median']), inplace = True)

    

    test = test.merge(tmp, how='left', on=['model', 'type', 'registration_year'])

    test[cat + '_mean'].fillna(np.nanmean(df[cat + '_mean']), inplace = True)

    test[cat + '_median'].fillna(np.nanmedian(df[cat + '_median']), inplace = True)

    test[cat].fillna(np.nanmedian(test[cat + '_median']), inplace = True)

def addlogs(my_df, ls):

    m = my_df.shape[1]

    for l in ls:

        my_df = my_df.assign(newcol=pd.Series(np.log(1.01+my_df[l])).values)   

        my_df.columns.values[m] = l + '_log'

        m += 1

    return my_df



def addSquared(my_df, ls):

    m = my_df.shape[1]

    for l in ls:

        my_df = my_df.assign(newcol=pd.Series(my_df[l]*my_df[l]).values)   

        my_df.columns.values[m] = l + '_sq'

        m += 1

    return my_df 



num_col_transf_list = ['engine_capacity','insurance_price',

                        'city_count', 'latitude_mean', 'longitude_mean', 'YearCar',

                        'insurance_price_mean', 'insurance_price_median',

                        'engine_capacity_mean', 'engine_capacity_median']





df = addlogs(df, num_col_transf_list)

df = addSquared(df, num_col_transf_list)

test = addlogs(test, num_col_transf_list)

test = addSquared(test, num_col_transf_list)
cat_col = ['type',  'gearbox', 'model', 'fuel', 'brand', 'damage', 'reg_year_str', 'zip_size', 'IsOld']



LE_mapper = {}

for category in cat_col:

    LE_mapper[category] = LabelEncoder()

    df[category] = LE_mapper[category].fit_transform(df[category])



for item in LE_mapper.items():  

    category = item[0]

    le = item[1]

    le_dict = dict(zip(le.classes_, le.transform(le.classes_)))

    test[category] = test[category].apply(lambda x: le_dict.get(x, -99999999))
# prep for modeling



X = df[df.columns.difference(['price'])]

y = df['price']

categorical_features = [i for i, e in enumerate(X.columns) if e in cat_col]
# create custom scoring



def mape(y_true, y_pred):

    y_pred[y_pred < 0 ] = 0

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100



mape_scorer = make_scorer(mape, greater_is_better = False)
# set diapason of parameters



lgbm_space = {

        'boosting_type': hp.choice('boosting_type', ['dart','gbdt','goss']),  

        'learning_rate': hp.choice('learning_rate', np.arange(0.005, 0.1005, 0.005)),

        'n_estimators': hp.choice('n_estimators', np.arange(100, 7001, 25, dtype=int)),

        'max_depth': hp.choice('max_depth', np.arange(5, 70, 2, dtype=int)),

        'num_leaves': hp.choice('num_leaves', [3,5,7,15,31,50,75,100]),

    

        'lambda_l1':  hp.loguniform('lambda_l1', -3, 2),

        'lambda_l2':  hp.loguniform('lambda_l2', -3, 2),

    

#         'num_leaves': hp.choice('num_leaves', np.arange(5, 31, 1, dtype=int)),    

#         'bagging_fraction': hp.uniform('bagging_fraction', 0, 1), 

#         'feature_fraction': hp.uniform('feature_fraction', 0, 1),   

#         'min_data_in_leaf': hp.choice('min_data_in_leaf', np.arange(2, 31, 1, dtype=int)),

#         'max_bin': hp.choice('max_bin', np.arange(200, 2000, 10, dtype=int)),

#         'bagging_freq': hp.choice('bagging_freq', np.arange(0, 11, 1, dtype=int)),

#         'min_child_weight': hp.uniform('min_child_weight', 0.1, 10),

#         'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.),

#         'subsample': hp.uniform('subsample', 0.5, 1.),

#         'bagging_fraction' : hp.uniform('bagging_fraction', 0.01, 1.0),

#         'feature_fraction' :  hp.uniform('feature_fraction', 0.5, 1.0),

#         'min_gain_to_split' : hp.uniform('min_gain_to_split', 0.0, 1.0),

    }
lgbm_model = lgb.LGBMRegressor(objective='regression', verbosity=-1, nthread=-1, random_state=42)



# model and fit params

params = dict(

        objective='regression',

        verbosity=-1,

        nthread=-1,

        random_state=42,

        categorical_feature=categorical_features,

)



fit_params = {

        'categorical_feature':categorical_features,

         }
def lgbm_objective(params):

    est = lgb.LGBMRegressor(verbosity=-1,nthread=-1,random_state=42,**params) 



    score = cross_val_score(

                est,

                X.values, y.values,

                scoring = mape_scorer,

                cv = KFold(5),

                n_jobs= -1,

                fit_params = fit_params,

    )

    print(abs(score))

    return abs(score.mean())

trials = Trials()
#  model_params = lgbm_model.get_params()

#  hp_lgbm_best = fmin(

#      fn=lgbm_objective,

#      space=lgbm_space,

#      algo=tpe.suggest,

#      max_evals=25,

#      trials=trials

#  )

#

#  best_params_lgbm = space_eval(lgbm_space, hp_lgbm_best)
filename = '../input/carsdata/model_lgbm.joblib'

model_lgbm = joblib.load(filename)
model_lgbm
test['pred'] = model_lgbm.predict(test)

test['pred'] = np.exp(test['pred']) - 1 



sample_submission['Predicted'] = test['pred']

sample_submission.to_csv('sample_submission.csv', index=False)