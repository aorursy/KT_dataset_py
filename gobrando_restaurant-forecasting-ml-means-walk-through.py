import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import ensemble, neighbors, linear_model, metrics, preprocessing
from datetime import datetime
import glob, re
data = {
    'ar': pd.read_csv('../input/air_reserve.csv'),
    'as': pd.read_csv('../input/air_store_info.csv'),
    'tra': pd.read_csv('../input/air_visit_data.csv'),
    'hol': pd.read_csv('../input/date_info.csv').rename(columns={'calendar_date': 'visit_date'}),
    'hr': pd.read_csv('../input/hpg_reserve.csv'),
    'hs': pd.read_csv('../input/hpg_store_info.csv'),
    'tes': pd.read_csv('../input/sample_submission.csv'),
    'id': pd.read_csv('../input/store_id_relation.csv')
}
data['id'].sample(3)
data['hr'] = data['hr'].merge(data['id'], how='inner', on=['hpg_store_id'])
data['hr'].sample(5)
# Notice datetime initially an object. We will fix that
np.dtype(data['hr']['visit_datetime'])
for df in ['ar', 'hr']:
    data[df]['visit_datetime'] = pd.to_datetime(data[df]['visit_datetime'])
    data[df]['visit_datetime'] = data[df]['visit_datetime'].dt.date
    data[df]['reserve_datetime'] = pd.to_datetime(data[df]['reserve_datetime'])
    data[df]['reserve_datetime'] = data[df]['reserve_datetime'].dt.date
    data[df]['reserve_datetime_diff'] = data[df].apply(
        lambda r: (r['visit_datetime'] - r['reserve_datetime']).days, axis=1)
    data[df] = data[df].groupby(['air_store_id', 'visit_datetime'], as_index=False)[['reserve_datetime_diff',
                                                                                     'reserve_visitors']].sum().rename(columns={'visit_datetime':'visit_date'})
data['hr'].sample(5)
data['ar'].sample(4)
data['as'].sample(5)
lbl = preprocessing.LabelEncoder()
data['as']['air_genre_name'] = lbl.fit_transform(data['as']['air_genre_name'])
data['as']['air_area_name'] = lbl.fit_transform(data['as']['air_area_name'])
data['as'].sample(3)
data['tra'].sample(4)
data['tra']['visit_date'] = pd.to_datetime(data['tra']['visit_date'])
data['tra']['day_of_week'] = data['tra']['visit_date'].dt.dayofweek
data['tra']['month'] = data['tra']['visit_date'].dt.month
data['tra']['year'] = data['tra']['visit_date'].dt.year
data['tra']['visit_date'] = data['tra']['visit_date'].dt.date

data['tra'].sample(4)
data['hol'].sample(5)
data['hol'] = data['hol'].drop(columns=['day_of_week'])
data['hol']['visit_date'] = pd.to_datetime(data['hol']['visit_date'])
data['hol']['visit_date'] = data['hol']['visit_date'].dt.date
data['hol'].sample(4)
data['hs'].sample(4)
data['tes'].sample(3)
data['tes']['visit_date'] = data['tes']['id'].map(lambda x: str(x).split('_')[2])
data['tes']['air_store_id'] = data['tes']['id'].map(lambda x: '_'.join(x.split('_')[:2]))
data['tes']['visit_date'] = pd.to_datetime(data['tes']['visit_date'])
data['tes']['day_of_week'] = data['tes']['visit_date'].dt.dayofweek
data['tes']['month'] = data['tes']['visit_date'].dt.month
data['tes']['year'] = data['tes']['visit_date'].dt.year
data['tes']['visit_date'] = data['tes']['visit_date'].dt.date
data['tes'].sample(4)
# Store each unique restaurant in an array
unique_stores = data['tes']['air_store_id'].unique()
# Break each restaurant into 7 rows to track each day of the week
stores = pd.concat([pd.DataFrame({'air_store_id': unique_stores, 
                                  'day_of_week': [i]*len(unique_stores)}) for i in range(7)], 
                   axis=0, ignore_index=True)
# Make a temporary variable to store new 'tra' features on visitors
# Then merge that new feature into our stores dataframe 
tmp = data['tra'].groupby(['air_store_id', 'day_of_week'], 
                          as_index=False)['visitors'].min().rename(columns={'visitors':'min_visitors'})
stores = stores.merge(tmp, how='left', on=['air_store_id','day_of_week'])
# Continue this process for mean, max, and count of visitors
tmp = data['tra'].groupby(['air_store_id', 'day_of_week'],
                         as_index=False)['visitors'].mean().rename(columns={'visitors':'mean_visitors'})
stores = stores.merge(tmp, how='left', on=['air_store_id', 'day_of_week'])
tmp = data['tra'].groupby(['air_store_id', 'day_of_week'], 
                          as_index=False)['visitors'].median().rename(columns={'visitors':'median_visitors'})
stores = stores.merge(tmp, how='left', on=['air_store_id', 'day_of_week'])
tmp = data['tra'].groupby(['air_store_id', 'day_of_week'], 
                          as_index=False)['visitors'].max().rename(columns={'visitors':'max_visitors'})
stores = stores.merge(tmp, how='left', on=['air_store_id', 'day_of_week'])
tmp = data['tra'].groupby(['air_store_id', 'day_of_week'], 
                          as_index=False)['visitors'].count().rename(columns={'visitors':'visitor count'})
stores = stores.merge(tmp, how='left', on=['air_store_id', 'day_of_week'])
# Now we'll merge 'as' in
stores = stores.merge(data['as'], how='left', on=['air_store_id'])
# Check everything looks good and is machine readable
stores.head(4)
train = pd.merge(data['tra'], data['hol'], how='left', on=['visit_date'])
train = pd.merge(data['tra'], stores, how='left', on=['air_store_id', 'day_of_week'])
test = pd.merge(data['tes'], data['hol'], how='left', on=['visit_date'])
test = pd.merge(data['tes'], stores, how='left', on=['air_store_id', 'day_of_week'])
test.sample(3)
data['ar'] = data['ar'].merge(data['hr'], how='left', on=['air_store_id', 'visit_date', 'reserve_datetime_diff', 'reserve_visitors'])
data['ar'].sample(4)
train = train.merge(data['ar'], how='left', on=['air_store_id', 'visit_date'])
test = test.merge(data['ar'], how='left', on=['air_store_id', 'visit_date'])
train.sample(5)
col = [c for c in train if c not in ['id', 'air_store_id', 'visit_date', 'visitors']]
train = train.fillna(-1)
test = test.fillna(-1)
train.sample(4)
def RMSLE(y, pred):
    return metrics.mean_squared_error(y, pred)**0.5
etc = ensemble.ExtraTreesRegressor(n_estimators=225, max_depth=5, n_jobs=-1, random_state=3)
knn = neighbors.KNeighborsRegressor(n_jobs=-1, n_neighbors=4)
etc.fit(train[col], np.log1p(train['visitors'].values))
knn.fit(train[col], np.log1p(train['visitors'].values))
test['visitors'] = (etc.predict(test[col]) / 2) + (knn.predict(test[col]) / 2)
test['visitors'] = np.expm1(test['visitors']).clip(lower=0.)
sub1 = test[['id', 'visitors']].copy()
# Create a glob expression for finding and reading all .csv files in the data warehouse
dfs = {re.search('/([^/\.]*)\.csv', fn).group(1):
      pd.read_csv(fn) for fn in glob.glob('../input/*.csv')}

# store the CSVs locally for quick access
for k, v in dfs.items(): locals()[k] = v
#date_info['weight'] = ((date_info.index + 1) / len(date_info))       # LB 0.509
#date_info['weight'] = ((date_info.index + 1) / len(date_info)) ** 2  # LB 0.503
#date_info['weight'] = ((date_info.index + 1) / len(date_info)) ** 3  # LB 0.500
#date_info['weight'] = ((date_info.index + 1) / len(date_info)) ** 4  # LB 0.498
date_info['weight'] = ((date_info.index + 1) / len(date_info)) ** 5 # LB 0.497
visit_data = air_visit_data.merge(date_info, left_on = 'visit_date', right_on = 'calendar_date', how = 'left')
visit_data.drop('calendar_date', axis=1, inplace = True)
visit_data['visitors'] = visit_data.visitors.map(pd.np.log1p)
visit_data.head()
wmean = lambda x:((x.weight * x.visitors).sum() / x.weight.sum())
visitors = visit_data.groupby(['air_store_id', 'day_of_week', 'holiday_flg']).apply(wmean).rename('visitors').reset_index()
visitors.head()
sample_submission['air_store_id'] = sample_submission.id.map(lambda x: '_'.join(x.split('_')[:-1]))
sample_submission['calendar_date'] = sample_submission.id.map(lambda x: x.split('_')[2])
sample_submission.drop('visitors', axis=1, inplace=True)
sample_submission = sample_submission.merge(date_info, on='calendar_date', how='left')
sample_submission = sample_submission.merge(visitors, on=[
    'air_store_id', 'day_of_week', 'holiday_flg'], how='left')
sample_submission.head()
sample_submission.visitors.isnull().sum()
missings = sample_submission.visitors.isnull()
sample_submission.loc[missings, 'visitors'] = sample_submission[missings].merge(
    visitors[visitors.holiday_flg == 0], on=('air_store_id', 'day_of_week'), 
    how='left')['visitors_y'].values
sample_submission.isnull().sum()
missings = sample_submission.visitors.isnull()
sample_submission.loc[missings, 'visitors'] = sample_submission[missings].merge(
    visitors[['air_store_id', 'visitors']].groupby('air_store_id').mean().reset_index(), 
    on='air_store_id', how='left')['visitors_y'].values
# Double check we filled all the missing values
sample_submission.isnull().sum()
sample_submission['visitors'] = sample_submission.visitors.map(pd.np.expm1)
sub2 = sample_submission[['id', 'visitors']].copy()
sub_merge = pd.merge(sub1, sub2, on='id', how='inner')
## Geometric Mean  
sub_merge['visitors'] = (sub_merge['visitors_x'] * sub_merge['visitors_y']) ** (1/2)
sub_merge[['id', 'visitors']].to_csv('sub_geo_mean.csv', index = False)
sub_merge[['id', 'visitors']].head()
## Harmonic Mean 
sub_merge['visitors'] = 2/(1/sub_merge['visitors_x'] + 1/sub_merge['visitors_y'])
sub_merge[['id', 'visitors']].to_csv('sub_hrm_mean.csv', index = False)
sub_merge[['id', 'visitors']].head()
