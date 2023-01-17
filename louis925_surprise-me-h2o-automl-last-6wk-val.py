# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input/val6wk"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
%%time
"""
Contributions from:
DSEverything - Mean Mix - Math, Geo, Harmonic (LB 0.493) 
https://www.kaggle.com/dongxu027/mean-mix-math-geo-harmonic-lb-0-493
JdPaletto - Surprised Yet? - Part2 - (LB: 0.503)
https://www.kaggle.com/jdpaletto/surprised-yet-part2-lb-0-503
hklee - weighted mean comparisons, LB 0.497, 1ST
https://www.kaggle.com/zeemeen/weighted-mean-comparisons-lb-0-497-1st

Also all comments for changes, encouragement, and forked scripts rock

Keep the Surprise Going
"""

import glob, re
import numpy as np
import pandas as pd
from sklearn import *
from datetime import datetime
from xgboost import XGBRegressor
import h2o
from h2o.automl import H2OAutoML
h2o.init()

data = {
    'tra': pd.read_csv('../input/val6wk/air_visit_data.csv'),
    'as': pd.read_csv('../input/val6wk/air_store_info.csv'),
    'hs': pd.read_csv('../input/val6wk/hpg_store_info.csv'),
    'ar': pd.read_csv('../input/val6wk/air_reserve.csv'),
    'hr': pd.read_csv('../input/val6wk/hpg_reserve.csv'),
    'id': pd.read_csv('../input/val6wk/store_id_relation.csv'),
    'tes': pd.read_csv('../input/val6wk/sample_submission.csv'),
    'hol': pd.read_csv('../input/val6wk/date_info.csv').rename(columns={'calendar_date':'visit_date'})
    }

data['tes']['visitors'] = 0

data['hr'] = pd.merge(data['hr'], data['id'], how='inner', on=['hpg_store_id'])

for df in ['ar','hr']:
    data[df]['visit_datetime'] = pd.to_datetime(data[df]['visit_datetime'])
    data[df]['visit_datetime'] = data[df]['visit_datetime'].dt.date
    data[df]['reserve_datetime'] = pd.to_datetime(data[df]['reserve_datetime'])
    data[df]['reserve_datetime'] = data[df]['reserve_datetime'].dt.date
    data[df]['reserve_datetime_diff'] = data[df].apply(lambda r: (r['visit_datetime'] - r['reserve_datetime']).days, axis=1)
    tmp1 = data[df].groupby(['air_store_id','visit_datetime'], as_index=False)[['reserve_datetime_diff', 'reserve_visitors']].sum().rename(columns={'visit_datetime':'visit_date', 'reserve_datetime_diff': 'rs1', 'reserve_visitors':'rv1'})
    tmp2 = data[df].groupby(['air_store_id','visit_datetime'], as_index=False)[['reserve_datetime_diff', 'reserve_visitors']].mean().rename(columns={'visit_datetime':'visit_date', 'reserve_datetime_diff': 'rs2', 'reserve_visitors':'rv2'})
    data[df] = pd.merge(tmp1, tmp2, how='inner', on=['air_store_id','visit_date'])

data['tra']['visit_date'] = pd.to_datetime(data['tra']['visit_date'])
data['tra']['dow'] = data['tra']['visit_date'].dt.dayofweek
data['tra']['year'] = data['tra']['visit_date'].dt.year
data['tra']['month'] = data['tra']['visit_date'].dt.month
data['tra']['visit_date'] = data['tra']['visit_date'].dt.date

data['tes']['visit_date'] = data['tes']['id'].map(lambda x: str(x).split('_')[2])
data['tes']['air_store_id'] = data['tes']['id'].map(lambda x: '_'.join(x.split('_')[:2]))
data['tes']['visit_date'] = pd.to_datetime(data['tes']['visit_date'])
data['tes']['dow'] = data['tes']['visit_date'].dt.dayofweek
data['tes']['year'] = data['tes']['visit_date'].dt.year
data['tes']['month'] = data['tes']['visit_date'].dt.month
data['tes']['visit_date'] = data['tes']['visit_date'].dt.date

unique_stores = data['tes']['air_store_id'].unique()
stores = pd.concat([pd.DataFrame({'air_store_id': unique_stores, 'dow': [i]*len(unique_stores)}) for i in range(7)], axis=0, ignore_index=True).reset_index(drop=True)

#sure it can be compressed...
tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].min().rename(columns={'visitors':'min_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow']) 
tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].mean().rename(columns={'visitors':'mean_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])
tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].median().rename(columns={'visitors':'median_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])
tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].max().rename(columns={'visitors':'max_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])
tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].count().rename(columns={'visitors':'count_observations'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow']) 

stores = pd.merge(stores, data['as'], how='left', on=['air_store_id']) 
# NEW FEATURES FROM Georgii Vyshnia
stores['air_genre_name'] = stores['air_genre_name'].map(lambda x: str(str(x).replace('/',' ')))
stores['air_area_name'] = stores['air_area_name'].map(lambda x: str(str(x).replace('-',' ')))
lbl = preprocessing.LabelEncoder()
for i in range(4):
    stores['air_genre_name'+str(i)] = lbl.fit_transform(stores['air_genre_name'].map(lambda x: str(str(x).split(' ')[i]) if len(str(x).split(' '))>i else ''))
    stores['air_area_name' +str(i)] = lbl.fit_transform(stores['air_area_name'].map(lambda x: str(str(x).split(' ')[i]) if len(str(x).split(' '))>i else ''))

stores['air_genre_name'] = lbl.fit_transform(stores['air_genre_name'])
stores['air_area_name'] = lbl.fit_transform(stores['air_area_name'])

data['hol']['visit_date'] = pd.to_datetime(data['hol']['visit_date'])
data['hol']['day_of_week'] = lbl.fit_transform(data['hol']['day_of_week'])
data['hol']['visit_date'] = data['hol']['visit_date'].dt.date
train = pd.merge(data['tra'], data['hol'], how='left', on=['visit_date']) 
test = pd.merge(data['tes'], data['hol'], how='left', on=['visit_date']) 

train = pd.merge(train, stores, how='left', on=['air_store_id','dow']) 
test = pd.merge(test, stores, how='left', on=['air_store_id','dow'])

for df in ['ar','hr']:
    train = pd.merge(train, data[df], how='left', on=['air_store_id','visit_date']) 
    test = pd.merge(test, data[df], how='left', on=['air_store_id','visit_date'])

train['id'] = train.apply(lambda r: '_'.join([str(r['air_store_id']), str(r['visit_date'])]), axis=1)

train['total_reserv_sum'] = train['rv1_x'] + train['rv1_y']
train['total_reserv_mean'] = (train['rv2_x'] + train['rv2_y']) / 2
train['total_reserv_dt_diff_mean'] = (train['rs2_x'] + train['rs2_y']) / 2

test['total_reserv_sum'] = test['rv1_x'] + test['rv1_y']
test['total_reserv_mean'] = (test['rv2_x'] + test['rv2_y']) / 2
test['total_reserv_dt_diff_mean'] = (test['rs2_x'] + test['rs2_y']) / 2

# NEW FEATURES FROM JMBULL
train['date_int'] = train['visit_date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)
test['date_int'] = test['visit_date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)
train['var_max_lat'] = train['latitude'].max() - train['latitude']
train['var_max_long'] = train['longitude'].max() - train['longitude']
test['var_max_lat'] = test['latitude'].max() - test['latitude']
test['var_max_long'] = test['longitude'].max() - test['longitude']

# NEW FEATURES FROM Georgii Vyshnia
train['lon_plus_lat'] = train['longitude'] + train['latitude'] 
test['lon_plus_lat'] = test['longitude'] + test['latitude']

lbl = preprocessing.LabelEncoder()
train['air_store_id2'] = lbl.fit_transform(train['air_store_id'])
test['air_store_id2'] = lbl.transform(test['air_store_id'])

train = train.fillna(-999)
test = test.fillna(-999)

train['visitors'] = np.log1p(train['visitors'].values)

print('Pre-processing done!')

htrain = h2o.H2OFrame(train)
htest = h2o.H2OFrame(test)

htrain.drop(['id', 'air_store_id', 'visit_date'])
htest.drop(['id', 'air_store_id', 'visit_date'])

x =htrain.columns
y ='visitors'
x.remove(y)

def RMSLE(y_, pred):
    return metrics.mean_squared_error(y_, pred)**0.5
%%time
print('Starting h2o autoML model!')  

aml = H2OAutoML(max_runtime_secs = 3350)
aml.train(x=x, y =y, training_frame=htrain, leaderboard_frame = htest)

print('Generate predictions...')
htrain.drop(['visitors'])
preds = aml.leader.predict(htrain)
preds = preds.as_data_frame()
print('RMSLE H2O automl leader: ', RMSLE(train['visitors'].values, preds))

preds = aml.leader.predict(htest)
preds = preds.as_data_frame()

test['visitors'] = preds
test['visitors'] = np.expm1(test['visitors']).clip(lower=0.)
sub1 = test[['id','visitors']].copy()
#del train; del data; del htrain; del htest;
sub1.to_csv('h2o_sub1.csv', index=False)
# from hklee
# https://www.kaggle.com/zeemeen/weighted-mean-comparisons-lb-0-497-1st/code
#dfs = { re.search('/([^/\.]*)\.csv', fn).group(1):
#    pd.read_csv(fn)for fn in glob.glob('../input/val6wk/*.csv')}

#for k, v in dfs.items(): locals()[k] = v

air_visit_data = pd.read_csv('../input/val6wk/air_visit_data.csv')
air_store_info = pd.read_csv('../input/val6wk/air_store_info.csv')
hpg_store_info = pd.read_csv('../input/val6wk/hpg_store_info.csv')
air_reserve = pd.read_csv('../input/val6wk/air_reserve.csv')
hpg_reserve = pd.read_csv('../input/val6wk/hpg_reserve.csv')
store_id_relation = pd.read_csv('../input/val6wk/store_id_relation.csv')
sample_submission = pd.read_csv('../input/val6wk/sample_submission.csv')
date_info = pd.read_csv('../input/val6wk/date_info.csv')
wkend_holidays = date_info.apply(
    (lambda x:(x.day_of_week=='Sunday' or x.day_of_week=='Saturday') and x.holiday_flg==1), axis=1)
date_info.loc[wkend_holidays, 'holiday_flg'] = 0
date_info['weight'] = ((date_info.index + 1) / len(date_info)) ** 5  

visit_data = air_visit_data.merge(date_info, left_on='visit_date', right_on='calendar_date', how='left')
visit_data.drop('calendar_date', axis=1, inplace=True)
visit_data['visitors'] = visit_data.visitors.map(pd.np.log1p)

wmean = lambda x:( (x.weight * x.visitors).sum() / x.weight.sum() )
visitors = visit_data.groupby(['air_store_id', 'day_of_week', 'holiday_flg']).apply(wmean).reset_index()
visitors.rename(columns={0:'visitors'}, inplace=True) # cumbersome, should be better ways.

sample_submission['air_store_id'] = sample_submission.id.map(lambda x: '_'.join(x.split('_')[:-1]))
sample_submission['calendar_date'] = sample_submission.id.map(lambda x: x.split('_')[2])
sample_submission.drop('visitors', axis=1, inplace=True)
sample_submission = sample_submission.merge(date_info, on='calendar_date', how='left')
sample_submission = sample_submission.merge(visitors, on=[
    'air_store_id', 'day_of_week', 'holiday_flg'], how='left')

missings = sample_submission.visitors.isnull()
sample_submission.loc[missings, 'visitors'] = sample_submission[missings].merge(
    visitors[visitors.holiday_flg==0], on=('air_store_id', 'day_of_week'), 
    how='left')['visitors_y'].values

missings = sample_submission.visitors.isnull()
sample_submission.loc[missings, 'visitors'] = sample_submission[missings].merge(
    visitors[['air_store_id', 'visitors']].groupby('air_store_id').mean().reset_index(), 
    on='air_store_id', how='left')['visitors_y'].values

sample_submission['visitors'] = sample_submission.visitors.map(pd.np.expm1)
sub2 = sample_submission[['id', 'visitors']].copy()
sub2.to_csv('h2o_sub2.csv', index=False)
sub_merge = pd.merge(sub1, sub2, on='id', how='inner')

sub_merge['visitors'] = (sub_merge['visitors_x'] + sub_merge['visitors_y']* 1.1)/2
sub_merge[['id', 'visitors']].to_csv('h2o_final_sub.csv', index=False)

print('Leaderboard : ', aml.leaderboard)

print(' H2O automl leader performace : ', aml.leader)
sub_true = pd.read_csv('../input/val6wk/sample_submission.csv')
(sub_true['id'] == sub_merge['id']).all()
def rmsle(y, y0):
    '''Root mean squared logarithmic error'''
    if type(y) != np.ndarray: y = np.array(y)
    if type(y0) != np.ndarray: y0 = np.array(y0)
    y = y.reshape(-1)
    y0 = y0.reshape(-1)
    if len(y) != len(y0): print("rmsle: lenths don't agree", len(y), 'vs', len(y0))
    if np.any(np.isnan(y)): 
        print("rmsle: y contains ", np.sum(np.isnan(y)), 'NaN')
        y = np.nan_to_num(y)
    if np.any(np.isnan(y0)):
        print("rmsle: y0 contains ", np.sum(np.isnan(y0)), 'NaN')
        y0 = np.nan_to_num(y0)
    
    return np.sqrt(np.mean(np.power(np.log1p(np.clip(y, 0, None))
           - np.log1p(np.clip(y0, 0, None)), 2)))
rmsle(sub_true['visitors'].values, sub_merge['visitors'].values)  # final result
rmsle(sub_true['visitors'].values, sub1['visitors'].values)  # The model
rmsle(sub_true['visitors'].values, sub2['visitors'].values)  # hklee
sub_merge['visitors'].isnull().sum(), sub1['visitors'].isnull().sum(), sub2['visitors'].isnull().sum()
from sklearn.metrics import mean_squared_error
# Double check the score
mean_squared_error(np.log1p(sub_true['visitors'].values), np.log1p(sub_merge['visitors'].values))**0.5
