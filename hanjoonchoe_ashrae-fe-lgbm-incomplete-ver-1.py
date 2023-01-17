# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



from pathlib import Path

import os

import os, gc

import random

import datetime



from tqdm import tqdm_notebook as tqdm



# matplotlib and seaborn for plotting

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import KFold, StratifiedKFold

from sklearn import preprocessing

import warnings

warnings.filterwarnings("ignore")
## Function to reduce the DF size

def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
def dataset_reader():

    list=['weather_test.csv'

          ,'weather_train.csv'

          ,'test.csv'

          ,'train.csv'

          ,'building_metadata.csv']

    input = Path('/kaggle/input/ashrae-energy-prediction')

    #list= [c for c in os.listdir(input)]

    

    wtest = pd.read_csv(input/list[0],parse_dates=['timestamp'])

    wtrain = pd.read_csv(input/list[1],parse_dates=['timestamp'])

    test = pd.read_csv(input/list[2],parse_dates=['timestamp'])

    train = pd.read_csv(input/list[3],parse_dates=['timestamp'])

    bmdata = pd.read_csv(input/list[4])



    train['is_train'] = 1

    test['is_train'] = 0

    

    # Concatenate and Merge

    full = pd.concat([train,test],sort=True,ignore_index = True)

    mean_mr = train.groupby('building_id').meter_reading.mean().reset_index()

    bmdata = bmdata.merge(mean_mr,on='building_id',how='left')

    wfull = pd.concat([wtrain,wtest],sort=True,ignore_index = True)

    return full,wfull,bmdata



full,wfull,bmdata = dataset_reader()
bmdata.groupby('primary_use').meter_reading.mean().sort_values().plot(kind='bar')
bmdata.groupby('primary_use').meter_reading.mean().sort_values().reset_index()
primary_use = {0: 'Religious worship',

  1: 'Warehouse/storage',

  2: 'Technology/science',

  3: 'Other',

  4: 'Retail',

  5: 'Parking',

  6: 'Lodging/residential',

  7: 'Manufacturing/industrial',

  8: 'Public services',

  9: 'Food sales and service',

  10: 'Entertainment/public assembly',

  11: 'Utility',

  12: 'Office',

  13: 'Healthcare',

  14: 'Services',

  15: 'Education'}

inv_map = {v: k for k, v in primary_use.items()}

def primary_use_encoding(x):

    

    for use in inv_map.keys():

        if use == x:

            return inv_map[use]

        

bmdata['pu_label'] = bmdata['primary_use'].apply(primary_use_encoding)
bmdata['log_sqf'] = np.log(bmdata.square_feet)

#from scipy import stats

#bmdata['bcx_sqf'] = stats.boxcox(bmdata.square_feet)
'''

full['diff_pp'] = full.loc[(~full.precip_depth_1_hr.isnull())&(full.dew_temperature.isnull() | full.air_temperature.isnull())].precip_depth_1_hr.apply(pp_encoding)

full['diff_cd'] = full.loc[(~full.cloud_coverage.isnull())&(full.dew_temperature.isnull() | full.air_temperature.isnull())].cloud_coverage.apply(cd_encoding)

def pp_encoding(x):

    

    if np.isnan(x):

        return x

    

    else:

        for diff in diff_precip['mean'].keys():



            if diff == x:

                return round(float(np.random.normal(diff_precip['mean'][diff], diff_precip['std'][diff], 1)),2)

        

    return x

def cd_encoding(x):

    

    if np.isnan(x):

        return x

    else:

        for diff in diff_cloud['mean'].keys():



            if diff == x:

                return round(float(np.random.normal(diff_cloud['mean'][diff], diff_cloud['std'][diff], 1)),2)

    return x

    

    full['est_dew_p'] = full.air_temperature.loc[~full.diff_pp.isnull()] - full.diff_pp

full['est_air_p'] = full.dew_temperature.loc[~full.diff_pp.isnull()] + full.diff_pp

full['est_dew_c'] = full.air_temperature.loc[~full.diff_cd.isnull()] - full.diff_cd

full['est_air_c'] = full.dew_temperature.loc[~full.diff_cd.isnull()] + full.diff_cd



#precipitation first

full.air_temperature.fillna(full.est_air_p, inplace=True)

full.dew_temperature.fillna(full.est_dew_p, inplace=True)

#cloud next

full.air_temperature.fillna(full.est_air_c, inplace=True)

full.dew_temperature.fillna(full.est_dew_c, inplace=True)

full['diff_pp'] = full.loc[(~full.precip_depth_1_hr.isnull())

                           &(full.dew_temperature.isnull() | full.air_temperature.isnull())].precip_depth_1_hr.apply(pp_encoding)

full['diff_cd'] = full.loc[(~full.cloud_coverage.isnull())

                           &(full.dew_temperature.isnull() | full.air_temperature.isnull())].cloud_coverage.apply(cd_encoding)



full['est_dew_p'] = full.air_temperature.loc[~full.diff_pp.isnull()] - full.diff_pp

full['est_air_p'] = full.dew_temperature.loc[~full.diff_pp.isnull()] + full.diff_pp

full['est_dew_c'] = full.air_temperature.loc[~full.diff_cd.isnull()] - full.diff_cd

full['est_air_c'] = full.dew_temperature.loc[~full.diff_cd.isnull()] + full.diff_cd



#precipitation first

full.air_temperature.fillna(full.est_air_p, inplace=True)

full.dew_temperature.fillna(full.est_dew_p, inplace=True)

#cloud next

full.air_temperature.fillna(full.est_air_c, inplace=True)

full.dew_temperature.fillna(full.est_dew_c, inplace=True)

'''

bmdata.groupby('primary_use').floor_count.mean().apply(np.ceil).plot(kind='bar')
floor_avg = bmdata.groupby('primary_use').floor_count.mean().apply(np.ceil).fillna(1).to_dict()
floor_avg = bmdata.groupby('primary_use').floor_count.mean().apply(np.ceil).fillna(1).to_dict()

def floor_encoding(x):

    

    if pd.isna(x):

        return np.nan

    else:

        for floor in floor_avg.keys():

            if floor in x:

                return floor_avg[floor]

    return np.nan



new = bmdata.loc[bmdata.floor_count.isnull()].primary_use.apply(floor_encoding)

bmdata['floor_count'].fillna(new, inplace=True)
from sklearn import preprocessing

encoder = LabelEncoder()

bmdata['primary_use'] = encoder.fit_transform(bmdata['primary_use'])
bmdata['year_built']=bmdata['year_built'].fillna(bmdata['year_built'].mean())

bmdata['year_built']=bmdata['year_built']-1900

bmdata.isnull().sum()
#from tqdm import tqdm

#lists = ['air_temperature','dew_temperature','cloud_coverage','sea_level_pressure','wind_direction','wind_speed','precip_depth_1_hr']

#size = full.building_id.nunique()

#for li in lists:

    #print(li)

    #for i in tqdm(range(size)):

        #full[li].update(full.loc[full.building_id==i][li].interpolate(method='pchip',limit_direction='both'))
from tqdm import tqdm

lists = ['air_temperature','dew_temperature','cloud_coverage','sea_level_pressure','wind_direction','wind_speed','precip_depth_1_hr']

size = wfull.site_id.nunique()

for li in lists:

    print(li)

    for i in tqdm(range(size)):

        wfull[li].update(wfull.loc[wfull.site_id==i][li].interpolate(method='ffill'))

        wfull[li].update(wfull.loc[wfull.site_id==i][li].interpolate(method='bfill'))

wfull.isnull().sum()
wfull.columns
wfull.groupby('site_id')[['air_temperature', 'cloud_coverage', 'dew_temperature',

       'precip_depth_1_hr', 'sea_level_pressure', 'site_id', 'timestamp',

       'wind_direction', 'wind_speed']].mean()
wfull['Month']= wfull.timestamp.dt.month

wfull['Day']= wfull.timestamp.dt.day

wfull['Hour'] = wfull.timestamp.dt.hour

wfull['Weekday'] = wfull.timestamp.dt.weekday
for i in tqdm(range(12)):

    wfull.update(wfull[wfull.Month==i+1].fillna(wfull[wfull.Month==i+1].mean()))
dt = wfull.dew_temperature

at = wfull.air_temperature

ws = wfull.wind_speed

#Relative Humidity

wfull['RH'] = 100*(0.6108*np.exp((17.27*dt)/(dt+237.3)))/(0.6108*np.exp((17.27*at)/(at+237.3)))
wfull.sea_level_pressure = wfull.sea_level_pressure - 1000

wfull.wind_direction = wfull.wind_direction%360

wfull.isnull().sum()
wfull.groupby('Month')[['air_temperature','dew_temperature','cloud_coverage','sea_level_pressure','wind_direction','wind_speed','precip_depth_1_hr']].mean()
import gc

gc.collect()
full = reduce_mem_usage(full)

wfull = reduce_mem_usage(wfull)

bmdata = reduce_mem_usage(bmdata)
del bmdata['meter_reading']

full = full.merge(bmdata, on='building_id', how='left')

full = full.merge(wfull, on=['site_id', 'timestamp'], how='left')

del bmdata

del wfull

gc.collect()
full.sample(5)
train = full.loc[(full.is_train==1)&(full.building_id<15)]

test = full[full.is_train==0]

print('done')
train.shape
train["meter_reading"]=np.log1p(train["meter_reading"])

print('done here')
target= 'meter_reading'

do_not_use = ['meter_reading'

                 ,'is_train'

                ,'row_id'

                ,'square_feet'

                ,'timestamp'

                ,'primary_use'

                ,'random']



feature_columns = [c for c in full.columns if c not in do_not_use ]

feature_columns
gc.collect()

gc.collect()
## Training(LGBM)
import lightgbm as lgb

folds = 4

seed = 777

models=[]

feature_importance = pd.DataFrame()

kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

for train_idx, valid_idx in tqdm(kf.split(train,train['building_id']),total=folds):

    print(f'Training and predicting for target {target}')

    Xtr = train[feature_columns].iloc[train_idx]

    Xv = train[feature_columns].iloc[valid_idx]

    ytr = train[target].iloc[train_idx].values

    yv = train[target].iloc[valid_idx].values

    print('Train_size: ',Xtr.shape[0],'Validation_size: ', ytr.shape[0])

    

    dtrain = lgb.Dataset(Xtr, label=ytr)

    dvalid = lgb.Dataset(Xv, label=yv)

    

    params = {

            'boosting_type': 'gbdt',

            'objective': 'regression',

            'metric': {'rmse'},

            'learning_rate': 0.5,

            'feature_fraction': 0.8,

            'bagging_fraction': 0.8,

            'bagging_freq' : 5

            }

    model = lgb.train(params,

                dtrain,

                num_boost_round=2000,

                valid_sets=(dtrain, dvalid),

               early_stopping_rounds=20,

               verbose_eval = 20)

    

    

    #feature importance

    #f_imp = pd.DataFrame()

    #f_imp['feature'] = feature_columns

    #f_imp["importance"] = model.feature_importances_

    #f_imp["fold"] = nfold

    #nfold += 1

    #feature_importance = pd.concat([feature_importance, f_imp],axis=0,ignore_index=True)

    models.append(model)

    gc.collect()
import matplotlib.pyplot as plt

feature_imp = pd.DataFrame(sorted(zip(models[0].feature_importance(), models[0].feature_name()),reverse = True), columns=['Value','Feature'])

plt.figure(figsize=(10, 5))

sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))

plt.title('LightGBM Features (avg over folds)')

plt.tight_layout()

plt.show()
# split test data into batches

set_size = len(test)

iterations = 50

batch_size = set_size // iterations



print(set_size, iterations, batch_size)

assert set_size == iterations * batch_size
meter_reading = []

for i in tqdm(range(iterations)):

    pos = i*batch_size

    fold_preds = [np.expm1(model.predict(test[feature_columns].iloc[pos : pos+batch_size])) for model in models]

    meter_reading.extend(np.mean(fold_preds, axis=0))



print(len(meter_reading))

assert len(meter_reading) == set_size
submission = pd.read_csv('/kaggle/input/ashrae-energy-prediction/sample_submission.csv')

submission['meter_reading'] = np.clip(meter_reading, a_min=0, a_max=None)
submission.to_csv('submission.csv', index=False)

submission.head(9)
#explainer = shap.TreeExplainer(models[0])

#shap_values = explainer.shap_values(train[feature_columns])
#shap.force_plot(explainer.expected_value,shap_values[0,:] ,train[feature_columns].iloc[0,:], matplotlib=True)
#shap.summary_plot(shap_values, train[feature_columns], plot_type="bar")

