# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import numpy as np, pandas as pd

import glob, re

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import AdaBoostClassifier

from sklearn.svm import SVC

from sklearn.grid_search import GridSearchCV



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

ora1= sample_submission

print('split date')

ora1['year'] = ora1.calendar_date.map(lambda x: (x.split('-')[0]))

ora1['month'] = ora1.calendar_date.map(lambda x: (x.split('-')[1]))

ora1['day'] = ora1.calendar_date.map(lambda x: (x.split('-')[2]))

ora1['mhend_flg'] = ora1.day.map(lambda x: 1 if int(x)>=25 else 0)





ora1=pd.get_dummies(ora1, columns = ['day_of_week'])

ora1=pd.get_dummies(ora1, columns = ['air_store_id'])



#ora1['air_store_id'] = ora1.air_store_id.map(lambda x: int( (x.split('_')[1]),16 ) )



ora1.head()
missings = ora1.visitors.isnull()

test = ora1[missings].copy()

test.drop(['visitors'], axis=1, inplace=True)

test.drop(['id'], axis=1, inplace=True)

test.drop(['calendar_date'], axis=1, inplace=True)



x = ora1[-missings].copy()

y = x['visitors'].copy()

x.drop(['visitors'], axis=1, inplace=True)

x.drop(['id'], axis=1, inplace=True)

x.drop(['calendar_date'], axis=1, inplace=True)



x.shape

# test.shape
#学習

model = RandomForestRegressor(max_depth = 1000,

                               max_features =834,

                               min_samples_split = 20,

                               n_estimators = 50,

                               n_jobs = -1,

                               random_state = 0)

model.fit(x, y)



#予測

output = model.predict(test)



#正答率

model.score(x, y)
#提出

ora1.loc[missings,'visitors'] =output

ora1['visitors'] = ora1.visitors.map(pd.np.expm1)



sub = pd.DataFrame({ 'id': ora1['id'],

                      'visitors':ora1['visitors']  })

sub.to_csv('random_result.csv', float_format='%.4f', index=None)

print("done")

ora1.head()