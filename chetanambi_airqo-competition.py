import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

import math

import gc

from tqdm import tqdm

from math import sqrt 

import lightgbm as lgb

from sklearn.metrics import mean_squared_error 

from sklearn.model_selection import KFold, train_test_split



pd.set_option("display.max_rows", 1000)

pd.set_option("display.max_columns", 1000)
train = pd.read_csv("/kaggle/input/zindi-airqo-competition/Train.csv")

test = pd.read_csv("/kaggle/input/zindi-airqo-competition/Test.csv")

sub = pd.read_csv('/kaggle/input/zindi-airqo-competition/sample_sub.csv')

meta_data = pd.read_csv('/kaggle/input/zindi-airqo-competition/airqo_metadata.csv', index_col=0)
train_copy = train.copy() 

test_copy = test.copy()
train.shape, test.shape
train.head(3)
test.head(3)
train['location'].value_counts() 
test['location'].value_counts() 
train.isnull().sum()
# covert features from string to List of values 

def replace_nan(x):

    if x == ' ':

        return np.nan

    else :

        return float(x)

    

features = ['temp','precip','rel_humidity','wind_dir','wind_spd','atmos_press']



for feature in features : 

    train_copy[feature] = train_copy[feature].apply(lambda x: [ replace_nan(X) for X in x.replace("nan"," ").split(",")])

    test_copy[feature] = test_copy[feature].apply(lambda x: [ replace_nan(X)  for X in x.replace("nan"," ").split(",")])    



df_copy = pd.concat([train_copy,test_copy],sort=False).reset_index(drop=True)

    

def aggregate_features(x,col_name):

    x['max_' + col_name] = x[col_name].apply(np.max)

    x['min_' + col_name] = x[col_name].apply(np.min)

    x['mean_' + col_name] = x[col_name].apply(np.mean)

    x['std_' + col_name] = x[col_name].apply(np.std)

    x['var_' + col_name] = x[col_name].apply(np.var)

    x['median_' + col_name] = x[col_name].apply(np.median)

    x['ptp_' + col_name] = x[col_name].apply(np.ptp)

    return x  



def remove_nan_values(x):

    return [e for e in x if not math.isnan(e)]



for col_name in tqdm(features):

    df_copy[col_name] = df_copy[col_name].apply(remove_nan_values)

    

for col_name in tqdm(features):

    df_copy = aggregate_features(df_copy, col_name)
df_copy.drop(features, axis=1, inplace=True)

df_copy.drop(['ID','location','target'], axis=1, inplace=True)

df_copy.head(3)
df = pd.concat([train,test],sort=False).reset_index(drop=True)

df.head(2)
df.shape
df.isnull().sum()
for col in tqdm(['temp', 'precip', 'rel_humidity', 'wind_dir', 'wind_spd', 'atmos_press']):

    df[[col + '_' + str(i) for i in range(121)]] = df[col].str.split(',', expand=True).astype('float32')
df.shape
meta_data
# Meta features 

meta_data.drop(['dist_motorway'], axis=1, inplace=True)

meta_data.drop(['hh','hh_cook_charcoal','hh_cook_firewood','hh_burn_waste'], axis=1, inplace=True) # feature imp 

meta_data.drop(['dist_trunk','dist_secondary','dist_tertiary','dist_unclassified','dist_residential'], axis=1, inplace=True) # feature imp 

df = pd.merge(df, meta_data, on='location', how='left')

df.shape
df.drop(['ID', 'temp', 'precip', 'rel_humidity', 'wind_dir', 'wind_spd', 'atmos_press'], axis=1, inplace=True)

df['location'] = df['location'].astype('category')

df.head(2)
df = pd.concat([df, df_copy], axis=1)

df.head(3)
# Start index positions:

# temp: 2, precip: 123, rel_humidity: 244, wind_dir: 365, wind_spd: 486, atmos_press: 607



for i in tqdm(range(2, 26)):

    df['mean_temp_hr_' + str(i-2)] = df.iloc[:,[i, i+24, i+48, i+72, i+96]].mean(axis=1)

    df['median_temp_hr_' + str(i-2)] = df.iloc[:,[i, i+24, i+48, i+72, i+96]].median(axis=1)

    df['min_temp_hr_' + str(i-2)] = df.iloc[:,[i, i+24, i+48, i+72, i+96]].min(axis=1)

    df['max_temp_hr_' + str(i-2)] = df.iloc[:,[i, i+24, i+48, i+72, i+96]].max(axis=1)

    #df['std_temp_hr_' + str(i-2)] = df.iloc[:,[i, i+24, i+48, i+72, i+96]].std(axis=1)

    #df['var_temp_hr_' + str(i-2)] = df.iloc[:,[i, i+24, i+48, i+72, i+96]].var(axis=1)

    #df['mad_temp_hr_' + str(i-2)] = df.iloc[:,[i, i+24, i+48, i+72, i+96]].mad(axis=1)

    #df['sem_temp_hr_' + str(i-2)] = df.iloc[:,[i, i+24, i+48, i+72, i+96]].sem(axis=1)

    #df['skew_temp_hr_' + str(i-2)] = df.iloc[:,[i, i+24, i+48, i+72, i+96]].skew(axis=1)

    #df['kurt_temp_hr_' + str(i-2)] = df.iloc[:,[i, i+24, i+48, i+72, i+96]].kurt(axis=1)



for i in tqdm(range(123, 147)):

    df['mean_precip_hr_' + str(i-123)] = df.iloc[:,[i, i+24, i+48, i+72, i+96]].mean(axis=1)

    df['median_precip_hr_' + str(i-123)] = df.iloc[:,[i, i+24, i+48, i+72, i+96]].median(axis=1)

    df['min_precip_hr_' + str(i-123)] = df.iloc[:,[i, i+24, i+48, i+72, i+96]].min(axis=1)

    df['max_precip_hr_' + str(i-123)] = df.iloc[:,[i, i+24, i+48, i+72, i+96]].max(axis=1)

    #df['std_precip_hr_' + str(i-123)] = df.iloc[:,[i, i+24, i+48, i+72, i+96]].std(axis=1)

    #df['var_precip_hr_' + str(i-123)] = df.iloc[:,[i, i+24, i+48, i+72, i+96]].var(axis=1)

    #df['mad_precip_hr_' + str(i-123)] = df.iloc[:,[i, i+24, i+48, i+72, i+96]].mad(axis=1)

    #df['sem_precip_hr_' + str(i-123)] = df.iloc[:,[i, i+24, i+48, i+72, i+96]].sem(axis=1)

    #df['skew_precip_hr_' + str(i-123)] = df.iloc[:,[i, i+24, i+48, i+72, i+96]].skew(axis=1)

    #df['kurt_precip_hr_' + str(i-123)] = df.iloc[:,[i, i+24, i+48, i+72, i+96]].kurt(axis=1)

    

for i in tqdm(range(244, 268)):

    df['mean_rel_humidity_hr_' + str(i-244)] = df.iloc[:,[i, i+24, i+48, i+72, i+96]].mean(axis=1)

    df['median_rel_humidity_hr_' + str(i-244)] = df.iloc[:,[i, i+24, i+48, i+72, i+96]].median(axis=1)

    df['min_rel_humidity_hr_' + str(i-244)] = df.iloc[:,[i, i+24, i+48, i+72, i+96]].min(axis=1)

    df['max_rel_humidity_hr_' + str(i-244)] = df.iloc[:,[i, i+24, i+48, i+72, i+96]].max(axis=1)

    #df['std_rel_humidity_hr_' + str(i-244)] = df.iloc[:,[i, i+24, i+48, i+72, i+96]].std(axis=1)

    #df['var_rel_humidity_hr_' + str(i-244)] = df.iloc[:,[i, i+24, i+48, i+72, i+96]].var(axis=1)

    #df['mad_rel_humidity_hr_' + str(i-244)] = df.iloc[:,[i, i+24, i+48, i+72, i+96]].mad(axis=1)

    #df['sem_rel_humidity_hr_' + str(i-244)] = df.iloc[:,[i, i+24, i+48, i+72, i+96]].sem(axis=1)

    #df['skew_rel_humidity_hr_' + str(i-244)] = df.iloc[:,[i, i+24, i+48, i+72, i+96]].skew(axis=1)

    #df['kurt_rel_humidity_hr_' + str(i-244)] = df.iloc[:,[i, i+24, i+48, i+72, i+96]].kurt(axis=1)

    

for i in tqdm(range(365, 389)):

    df['mean_wind_dir_hr_' + str(i-365)] = df.iloc[:,[i, i+24, i+48, i+72, i+96]].mean(axis=1)

    df['median_wind_dir_hr_' + str(i-365)] = df.iloc[:,[i, i+24, i+48, i+72, i+96]].median(axis=1)

    df['min_wind_dir_hr_' + str(i-365)] = df.iloc[:,[i, i+24, i+48, i+72, i+96]].min(axis=1)

    df['max_wind_dir_hr_' + str(i-365)] = df.iloc[:,[i, i+24, i+48, i+72, i+96]].max(axis=1)

    #df['std_wind_dir_hr_' + str(i-365)] = df.iloc[:,[i, i+24, i+48, i+72, i+96]].std(axis=1)

    #df['var_wind_dir_hr_' + str(i-365)] = df.iloc[:,[i, i+24, i+48, i+72, i+96]].var(axis=1)

    #df['mad_wind_dir_hr_' + str(i-365)] = df.iloc[:,[i, i+24, i+48, i+72, i+96]].mad(axis=1)

    #df['sem_wind_dir_hr_' + str(i-365)] = df.iloc[:,[i, i+24, i+48, i+72, i+96]].sem(axis=1)

    #df['skew_wind_dir_hr_' + str(i-365)] = df.iloc[:,[i, i+24, i+48, i+72, i+96]].skew(axis=1)

    #df['kurt_wind_dir_hr_' + str(i-365)] = df.iloc[:,[i, i+24, i+48, i+72, i+96]].kurt(axis=1)

    

for i in tqdm(range(486, 510)):

    df['mean_wind_spd_hr_' + str(i-486)] = df.iloc[:,[i, i+24, i+48, i+72, i+96]].mean(axis=1)

    df['median_wind_spd_hr_' + str(i-486)] = df.iloc[:,[i, i+24, i+48, i+72, i+96]].median(axis=1)

    df['min_wind_spd_hr_' + str(i-486)] = df.iloc[:,[i, i+24, i+48, i+72, i+96]].min(axis=1)

    df['max_wind_spd_hr_' + str(i-486)] = df.iloc[:,[i, i+24, i+48, i+72, i+96]].max(axis=1)

    #df['std_wind_spd_hr_' + str(i-486)] = df.iloc[:,[i, i+24, i+48, i+72, i+96]].std(axis=1)

    #df['var_wind_spd_hr_' + str(i-486)] = df.iloc[:,[i, i+24, i+48, i+72, i+96]].var(axis=1)

    #df['mad_wind_spd_hr_' + str(i-486)] = df.iloc[:,[i, i+24, i+48, i+72, i+96]].mad(axis=1)

    #df['sem_wind_spd_hr_' + str(i-486)] = df.iloc[:,[i, i+24, i+48, i+72, i+96]].sem(axis=1)

    #df['skew_wind_spd_hr_' + str(i-486)] = df.iloc[:,[i, i+24, i+48, i+72, i+96]].skew(axis=1)

    #df['kurt_wind_spd_hr_' + str(i-486)] = df.iloc[:,[i, i+24, i+48, i+72, i+96]].kurt(axis=1)



for i in tqdm(range(607, 631)):

    df['mean_atmos_press_hr_' + str(i-607)] = df.iloc[:,[i, i+24, i+48, i+72, i+96]].mean(axis=1)

    df['median_atmos_press_hr_' + str(i-607)] = df.iloc[:,[i, i+24, i+48, i+72, i+96]].median(axis=1)

    df['min_atmos_press_hr_' + str(i-607)] = df.iloc[:,[i, i+24, i+48, i+72, i+96]].min(axis=1)

    df['max_atmos_press_hr_' + str(i-607)] = df.iloc[:,[i, i+24, i+48, i+72, i+96]].max(axis=1)

    #df['std_atmos_press_hr_' + str(i-607)] = df.iloc[:,[i, i+24, i+48, i+72, i+96]].std(axis=1)

    #df['var_atmos_press_hr_' + str(i-607)] = df.iloc[:,[i, i+24, i+48, i+72, i+96]].var(axis=1)

    #df['mad_atmos_press_hr_' + str(i-607)] = df.iloc[:,[i, i+24, i+48, i+72, i+96]].mad(axis=1)

    #df['sem_atmos_press_hr_' + str(i-607)] = df.iloc[:,[i, i+24, i+48, i+72, i+96]].sem(axis=1)

    #df['skew_atmos_press_hr_' + str(i-607)] = df.iloc[:,[i, i+24, i+48, i+72, i+96]].skew(axis=1)

    #df['kurt_atmos_press_hr_' + str(i-607)] = df.iloc[:,[i, i+24, i+48, i+72, i+96]].kurt(axis=1)
train = df[df.target.notnull()].reset_index(drop=True)

test = df[df.target.isna()].reset_index(drop=True)

test.drop('target', axis=1, inplace=True)
train.shape, test.shape
#del df_copy, train_copy, test_copy  

gc.collect()
X = train.drop(labels=['target'], axis=1)

y = train['target'].values
X.shape
X.head(3)
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=1)
X_train.shape, y_train.shape, X_valid.shape, y_valid.shape
train_data = lgb.Dataset(X_train, label=y_train)

test_data = lgb.Dataset(X_valid, label=y_valid)



param = {'objective': 'regression',

         'boosting': 'gbdt',  

         'metric': 'rmse',

         'learning_rate': 0.1, 

         'num_iterations': 750,

         'max_depth': -1,

         'min_data_in_leaf': 20,

         'bagging_fraction': 0.8,

         'bagging_freq': 1,

         'feature_fraction': 0.8

         }



clf = lgb.train(params=param, 

                early_stopping_rounds=50,

                verbose_eval=100,

                train_set=train_data,

                valid_sets=[test_data])



y_pred = clf.predict(X_valid) 
y_pred
np.sqrt(mean_squared_error(y_valid, y_pred)) # 26.365619764973086
pd.DataFrame(sorted(zip(clf.feature_importance(), X.columns), reverse=True), columns=['Value','Feature']).tail(150)
Xtest = test
Xtest.head(2)
from sklearn.model_selection import KFold



errlgb = []

y_pred_totlgb = []

i = 1



fold = KFold(n_splits=10, shuffle=True, random_state=42)



for train_index, test_index in fold.split(X):

    

    X_train, X_test = X.loc[train_index], X.loc[test_index]

    y_train, y_test = y[train_index], y[test_index]

    

    train_data = lgb.Dataset(X_train, label=y_train)

    test_data = lgb.Dataset(X_test, label=y_test)

    

    clf = lgb.train(params=param, 

                     early_stopping_rounds=100,

                     verbose_eval=100,

                     train_set=train_data,

                     valid_sets=[test_data])



    y_pred = clf.predict(X_test) 



    print("RMSE: ", np.sqrt(mean_squared_error(y_test, y_pred)))

    

    errlgb.append(np.sqrt(mean_squared_error(y_test, y_pred)))

    

    p = clf.predict(Xtest)

    

    y_pred_totlgb.append(p)

    

    print(f'-----------fold {i} completed------------')

    i+=1 
np.mean(y_pred_totlgb,0)
np.mean(errlgb, 0) # 24.525506793909283
y_pred = np.mean(y_pred_totlgb,0)
test = pd.read_csv("/kaggle/input/zindi-airqo-competition/Test.csv")

sub['ID'] = test['ID'].values

sub['target'] = y_pred
sub.head()
sub.to_csv('Output.csv', index=False)