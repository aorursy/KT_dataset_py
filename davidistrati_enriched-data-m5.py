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
calendar = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')

train = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')

black_friday_dates = ['2011-11-15', '2012-11-23', '2013-11-22', '2014-11-28','2015-11-27', '2016-11-25' ]

cyber_monday_dates = ['2011-11-28', '2012-11-26', '2013-12-02', '2014-12-01','2015-11-30', '2016-11-28' ]

holidays = ['OrthodoxEaster', 'IndependenceDay', 'Halloween', 'Thanksgiving', 'Christmas', 'NewYear', 'OrthodoxChristmas', 'OrthodoxChristmas', 'Easter', 'OrthodoxEaster']
train = train.assign(pop_in_mil=39.56)

train.loc[train['state_id'] == 'CA', 'pop_in_mil'] = 39.56

train.loc[train['state_id'] == 'TX', 'pop_in_mil'] = 28.7

train.loc[train['state_id'] == 'WI', 'pop_in_mil'] = 5.814



train = train.assign(avg_income_in_thousands=63.783)

train.loc[train['state_id'] == 'CA', 'avg_income_in_thousands'] = 63.783

train.loc[train['state_id'] == 'TX', 'avg_income_in_thousands'] = 60.629

train.loc[train['state_id'] == 'WI', 'avg_income_in_thousands'] = 77.687
black_friday_days = calendar[calendar['date'].isin(black_friday_dates)]['d'].values.tolist()

cyber_monday_days = calendar[calendar['date'].isin(cyber_monday_dates)]['d'].values.tolist()

holiday_days = calendar[calendar['event_name_1'].isin(holidays)]['d'].values.tolist()

december_january_days = calendar[pd.to_datetime(calendar['date']).dt.month.isin([12, 1])]['d'].values.tolist()



elemn_list = [black_friday_days, cyber_monday_days, holiday_days, december_january_days]



for k in elemn_list:

    for i in k:

        if (i in train.columns) == False:

            k.remove(i)

            

bf_cm_days = black_friday_days + cyber_monday_days 
is_holiday = pd.DataFrame([np.zeros(len(train.columns))], columns=train.columns)

is_holiday[holiday_days] = 1

is_bf_cm = pd.DataFrame([np.zeros(len(train.columns))], columns=train.columns)

is_bf_cm[bf_cm_days] = 1

is_december_or_january = pd.DataFrame([np.zeros(len(train.columns))], columns=train.columns)

is_december_or_january[december_january_days] = 1



is_holiday['id'] = 'is_holiday'

is_bf_cm['id'] = 'is_black_friday_cyber_monday'

is_december_or_january['id'] = 'is_december_or_january'
df = pd.concat([is_holiday, is_bf_cm, is_december_or_january, train], ignore_index=True)
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

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))



    return df
from sys import getsizeof

getsizeof(df)
reduce_mem_usage(df, verbose=True)
filename = 'enriched_data.csv'

df.to_csv(filename,index=False)