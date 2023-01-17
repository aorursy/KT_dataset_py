# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd

import numpy as np

from datetime import datetime

import xgboost as xgb

from xgboost import XGBRegressor, XGBClassifier, plot_importance

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_squared_error

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

air_reserve = pd.read_csv('../input/air_reserve.csv')

air_store_info= pd.read_csv('../input/air_store_info.csv')

air_visit_data = pd.read_csv('../input/air_visit_data.csv')

date_info = pd.read_csv('../input/date_info.csv')

hpg_store_info = pd.read_csv('../input/hpg_store_info.csv')

store_id_relation = pd.read_csv('../input/store_id_relation.csv')

sample_submission = pd.read_csv('../input/sample_submission.csv')

hpg_reserve = pd.read_csv('../input/hpg_reserve.csv')



air_combined = pd.merge(air_reserve, air_store_info, on='air_store_id', how='outer')

hpg_combined = pd.merge(hpg_reserve, hpg_store_info, on='hpg_store_id', how='left')



df = store_id_relation.merge(hpg_combined, on='hpg_store_id', how='left')

df2 = air_combined.merge(df, on='air_store_id', how='left')
master_col = df2[['air_store_id']]

df2.head(5)
print(df2.info())

print(len(df2['air_store_id'].value_counts()))
date_info['calendar_date'] = pd.to_datetime(date_info['calendar_date']).dt.date.astype(str)

#date_info['holiday_flg'] = date_info['holiday_flg'].map({1: 'Yes', 0: 'No'})

date_info['MTWTF'] = date_info['day_of_week'].map({'Monday': 1, 

                                                   'Tuesday': 2, 

                                                   'Wednesday': 3, 

                                                   'Thursday': 4, 

                                                   'Friday': 5,

                                                   'Saturday': 6, 

                                                   'Sunday': 7})



date_info['weekend_or_weekday'] = date_info['day_of_week'].map({'Monday': 0, 

                                                                'Tuesday': 0, 

                                                                'Wednesday': 0, 

                                                                'Thursday': 0, 

                                                                'Friday': 0,

                                                                'Saturday': 1, 

                                                                'Sunday': 1})

date_info2 = date_info.drop(['day_of_week'], axis=1)

date_info2.head(5)
sub_store = sample_submission['id'].apply(lambda x: str(x).split('_', 2)[:2])

sub_dates = pd.to_datetime(sample_submission['id'].apply(lambda x: str(x).split('_', 2)[2]).rename('Date'))

sub_stores = pd.Series(['_'.join(x) for x in sub_store]).rename('air_store_id')



sub_dt = pd.DataFrame({

        'air_store_id': sub_stores,

        'date': sub_dates.dt.date.astype(str),

        #'year': sub_dates.dt.year,

        'month': sub_dates.dt.month,

        'day': sub_dates.dt.day })



sub_df = pd.concat([sample_submission, sub_dt], axis=1)

sub_df2 = sub_df.merge(date_info2, left_on= sub_df['date'], right_on=date_info['calendar_date'])

sub_df3 = sub_df2.drop(['id', 'calendar_date', 'date'], axis =1)

sub_df3.head(5)
a = pd.to_datetime(df2['visit_datetime_x'])

#b = pd.to_datetime(df2['reserve_datetime_x'])

#c = pd.to_datetime(df2['visit_datetime_y'])

#d = pd.to_datetime(df2['reserve_datetime_y'])



datetime_df =pd.DataFrame({

        'air_visit_date': a.dt.date.astype(str),

        #'year': a.dt.year,

        'month': a.dt.month,

        'day': a.dt.day,

        

        #'air_visit_hour': a.dt.hour,

        #'air_visit_date': a.dt.date,

        #'air_visit_year': b.dt.year,

        

        #'air_reserve_month': b.dt.month,

        #'air_reserve_day': b.dt.day,

        #'air_reserve_hour': b.dt.hour,

         

        #'hpg_visit_date': c.dt.date,

        #'hpg_visit_year': c.dt.year,

        #'hpg_visit_month': c.dt.month,

        #'hpg_visit_day': c.dt.day,

        #'hpg_visit_hour': c.dt.hour,

        

        #'hpg_reserve_date': d.dt.date,

        #'hpg_reserve_year': d.dt.year,

        #'hpg_reserve_month': d.dt.month,

        #'hpg_reserve_day': d.dt.day,

        #'hpg_reserve_hour': d.dt.hour        

    }).fillna(0)



datetime_df2 = datetime_df.merge(date_info, left_on=datetime_df['air_visit_date'], 

                                 right_on=date_info['calendar_date'])
lbl = LabelEncoder()

categorical_df = pd.concat([master_col, df2[['air_genre_name','air_area_name','latitude_x','longitude_x',

                      'hpg_genre_name','hpg_area_name']].fillna('None_Stated').apply(lbl.fit_transform)], axis=1)



#categorical_dummy = pd.get_dummies(df2[['air_genre_name','air_area_name','latitude','longitude', 'hpg_genre_name','hpg_area_name']].fillna('None_Stated'))



sub_df4 = sub_df3.merge(categorical_df, left_on=sub_df3['air_store_id'], right_on=categorical_df['air_store_id'])  



combined_visitors = pd.Series(df2['reserve_visitors_x'].fillna(0) + df2['reserve_visitors_y'].fillna(0)).rename('visitors')





train_df = pd.concat([datetime_df2, combined_visitors, categorical_df], axis=1)
lbl= LabelEncoder()

ids = lbl.fit_transform(feats['air_store_id'].append(sub_df4['air_store_id_x']))

train_df['air_store_id_num'] = pd.Series(ids[:1171046])

sub_df4['air_store_id_num'] = pd.Series(ids[1171047:])



train_df2 = train_df.drop(['air_store_id', 'air_visit_date', 'calendar_date', 'day_of_week'], axis=1)

sub_df5 = sub_df4.drop(['air_store_id_x', 'air_store_id_y'], axis=1).reindex(columns=list(train_df2.columns.values))
vis_store = train_df2['visitors'].groupby(train_df2['air_store_id_num']).describe()

vis_dow = train_df2['visitors'].groupby(train_df2['MTWTF']).describe()

vis_store_dow = train_df2['visitors'].groupby([train_df2['air_store_id_num'], train_df2['MTWTF']]).describe()
vis_store_dow.head(15)
X_train, X_test, y_train, y_test = train_test_split(train_df2, train_df2['visitors'], test_size=0.1, random_state=7)



xgbR = XGBRegressor(learning_rate=0.1,

                   objective='reg:linear')

xgbR.fit(X_train, y_train)



def rmsle(h, y): 

    """

    Compute the Root Mean Squared Log Error for hypthesis h and targets y



    Args:

        h - numpy array containing predictions with shape (n_samples, n_targets)

        y - numpy array containing targets with shape (n_samples, n_targets)

    """

    return np.sqrt(np.square(np.log(h + 1) - np.log(y + 1)).mean())





pred = xgbR.predict(X_test)

mse = rmsle(pred, y_test)

print(mse)