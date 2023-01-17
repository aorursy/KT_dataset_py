# Benchmark

# log_transform-->weightmean-->0.497

# weightmean-->log_transfrom -->0.515

# 2017 data only log_transform -->weightmean -->0.497

# 2017 data only weightmean -->log_transform -->0.514
## Clone from https://www.kaggle.com/zusmani/baseline-lb-0-497?scriptVersionId=1837246



import numpy as np, pandas as pd

import glob, re



dfs = { re.search('/([^/\.]*)\.csv', fn).group(1):pd.read_csv(fn) for fn in glob.glob('../input/*.csv')}

print('data frames read:{}'.format(list(dfs.keys())))



print('local variables with the same names are created.')

for k, v in dfs.items(): locals()[k] = v



print('holidays at weekends are not special, right?')

wkend_holidays = date_info.apply((lambda x:(x.day_of_week=='Sunday' or x.day_of_week=='Saturday') and x.holiday_flg==1), axis=1)

date_info.loc[wkend_holidays, 'holiday_flg'] = 0



print('add decreasing weights from now')

date_info['weight'] = ((date_info.index + 1) / len(date_info)) ** 5  # LB 0.497
print('weighted mean visitors for each (air_store_id, day_of_week, holiday_flag) or (air_store_id, day_of_week)')

visit_data = air_visit_data.merge(date_info, left_on='visit_date', right_on='calendar_date', how='left')

visit_data.drop('calendar_date', axis=1, inplace=True)
visit_data['visit_date'] = pd.to_datetime(visit_data['visit_date'])
# train_data = visit_data[visit_data['visit_date']<'2017-04-01']

test_data = visit_data[visit_data['visit_date']>='2017-04-01']
train_data = visit_data[visit_data['visit_date'] >='2017-03-01']
test_data.head()
y_train_data = visit_data.loc[visit_data['visit_date']<'2017-04-01','visitors']

y_test_data = visit_data.loc[visit_data['visit_date']>='2017-04-01','visitors']

train_data.shape, y_train_data.shape, test_data.shape, y_test_data.shape
visit_data.tail()


train_data['visitors'] = train_data.visitors.map(pd.np.log1p)

wmean = lambda x:( (x.weight * x.visitors).sum() / x.weight.sum() )

visitors = train_data.groupby(['air_store_id', 'day_of_week', 'holiday_flg']).apply(wmean).reset_index()

visitors.rename(columns={0:'visitors'}, inplace=True) # cumbersome, should be better ways.
test_data.columns
test_data['visitors_predict'] = test_data.merge(visitors[visitors.holiday_flg==0], \

                            on=('air_store_id', 'day_of_week'), how='left')['visitors_y']
missings = test_data.visitors_predict.isnull()

test_data.loc[missings, 'visitors_predict']  = test_data[missings].merge(

visitors[['air_store_id', 'visitors']].groupby('air_store_id').\

    mean().reset_index(), on='air_store_id', how='left')['visitors_y'].values

test_data['visitors'] = test_data['visitors'].map(np.log1p)
test_data['air_store_id'].isin(train_data['air_store_id']).sum()
test_data[test_data['visitors_predict'].isnull()]
import sklearn.metrics
test_data[['air_store_id','day_of_week','visit_date','visitors','visitors_predict']].tail(10)
sklearn.metrics.mean_squared_error(

  np.array([2.3]),np.array([3]))
sklearn.metrics.mean_squared_error(\

            test_data.loc[~test_data.visitors_predict.isnull(),'visitors']

          ,test_data.loc[~test_data.visitors_predict.isnull(),'visitors_predict'])
sklearn.metrics.mean_squared_error(\

            test_data['visitors']

          ,test_data['visitors_predict'])
print('prepare to merge with date_info and visitors')

sample_submission['air_store_id'] = sample_submission.id.map(lambda x: '_'.join(x.split('_')[:-1]))

sample_submission['calendar_date'] = sample_submission.id.map(lambda x: x.split('_')[2])

sample_submission.drop('visitors', axis=1, inplace=True)

sample_submission = sample_submission.merge(date_info, on='calendar_date', how='left')

sample_submission = sample_submission.merge(visitors, on=['air_store_id', 'day_of_week', 'holiday_flg'], how='left')



# fill missings with (air_store_id, day_of_week)

missings = sample_submission.visitors.isnull()

sample_submission.loc[missings, 'visitors'] = sample_submission[missings].merge(

    visitors[visitors.holiday_flg==0], on=('air_store_id', 'day_of_week'), how='left')['visitors_y'].values



# fill missings with (air_store_id)

missings = sample_submission.visitors.isnull()

sample_submission.loc[missings, 'visitors'] = sample_submission[missings].merge(

    visitors[['air_store_id', 'visitors']].groupby('air_store_id').mean().reset_index(), on='air_store_id', how='left')['visitors_y'].values

    

sample_submission['visitors'] = sample_submission.visitors.map(pd.np.expm1)



sample_submission[['id', 'visitors']].to_csv('dumb_result.csv', float_format='%.4f', index=None)

print("done")
visitors