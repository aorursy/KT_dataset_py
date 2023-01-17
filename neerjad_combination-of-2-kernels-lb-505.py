# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

# print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
test_df = pd.read_csv('../input/recruit-restaurant-visitor-forecasting/sample_submission.csv')

air_data = pd.read_csv('../input/recruit-restaurant-visitor-forecasting/air_visit_data.csv', parse_dates=['visit_date'])

air_data.head()
air_data.shape, test_df.shape
#Add a column for day of the week

air_data['dow'] = air_data['visit_date'].dt.dayofweek

air_data.head()
test_df['store_id'], test_df['visit_date'] = test_df['id'].str[:20], test_df['id'].str[21:]

test_df.drop(['visitors'], axis=1, inplace=True)

test_df['visit_date'] = pd.to_datetime(test_df['visit_date'])
air_data.head()

air_data['visit_date'].min(), air_data['visit_date'].max()
#Use data only after 2017-01-28:

train = air_data[air_data['visit_date'] > '2017-01-28'].reset_index()

train['dow'] = train['visit_date'].dt.dayofweek

test_df['dow'] = test_df['visit_date'].dt.dayofweek

aggregation = {'visitors' :{'total_visitors' : 'median'}}



# Group by id and day of week - Median of the visitors is taken

agg_data = train.groupby(['air_store_id', 'dow']).agg(aggregation).reset_index()

agg_data.columns = ['air_store_id','dow','visitors']

agg_data['visitors'] = agg_data['visitors']
# Create the first intermediate submission file:

merged = pd.merge(test_df, agg_data, how='left', left_on=['store_id','dow'], right_on=['air_store_id','dow'])

final = merged[['id','visitors']]

final.fillna(0, inplace=True)
# Taken from the kernel - https://www.kaggle.com/zeemeen/weighted-mean-running-10-sec-lb-0-509

import glob, re



dfs = { re.search('/([^/\.]*)\.csv', fn).group(1):pd.read_csv(fn) for fn in glob.glob('../input/recruit-restaurant-visitor-forecasting/*.csv')}

print('data frames read:{}'.format(list(dfs.keys())))
print('local variables with the same names are created.')

for k, v in dfs.items(): locals()[k] = v



print('holidays at weekends are not special, right?')

wkend_holidays = date_info.apply((lambda x:(x.day_of_week=='Sunday' or x.day_of_week=='Saturday') and x.holiday_flg==1), axis=1)

date_info.loc[wkend_holidays, 'holiday_flg'] = 0



print('add decreasing weights from now')

date_info['weight'] = (date_info.index + 1) / len(date_info) 



print('weighted mean visitors for each (air_store_id, day_of_week, holiday_flag) or (air_store_id, day_of_week)')

visit_data = air_visit_data.merge(date_info, left_on='visit_date', right_on='calendar_date', how='left')

visit_data.drop('calendar_date', axis=1, inplace=True)

visit_data['visitors'] = visit_data.visitors.map(pd.np.log1p)



wmean = lambda x:( (x.weight * x.visitors).sum() / x.weight.sum() )

visitors = visit_data.groupby(['air_store_id', 'day_of_week', 'holiday_flg']).apply(wmean).reset_index()

visitors.rename(columns={0:'visitors'}, inplace=True) # cumbersome, should be better ways.



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



sample_submission = sample_submission[['id', 'visitors']]
len(final['visitors'][final['visitors'] ==0])
final['visitors'][final['visitors'] ==0] = sample_submission['visitors'][final['visitors'] ==0]
final['visitors'] = np.mean([final['visitors'], sample_submission['visitors']], axis = 0)
final.head()
final.to_csv('submission_kernel_mean.csv', index=False)