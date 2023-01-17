import pandas as pd

from IPython.display import display

pd.set_option('display.float_format', lambda x: '%.5f' % x)

air_reserve = pd.read_csv('../input/air_reserve.csv',parse_dates=['visit_datetime','reserve_datetime'])



hpg_reserve = pd.read_csv('../input/hpg_reserve.csv',parse_dates=['visit_datetime','reserve_datetime'])



air_store_info = pd.read_csv('../input/air_store_info.csv')



hpg_store_info = pd.read_csv('../input/hpg_store_info.csv')



store_relation = pd.read_csv('../input/store_id_relation.csv')



date_info = pd.read_csv('../input/date_info.csv',parse_dates=['calendar_date'])



air_visit = pd.read_csv('../input/air_visit_data.csv',parse_dates=['visit_date'])



sample_submission = pd.read_csv('../input/sample_submission.csv')
air_reserve.head(2) #all the air stores booking data
hpg_reserve.head(2) #all the hpg stores booking data
air_store_info.head(2) #description of air stores
hpg_store_info.head(2) #description of hpg stores
air_visit.head(2) #historical visits for air stores
date_info.head(2) 
store_relation.head(2) #hpg store to air store mapping
sample_submission.head(2) #air id and date is merged together
#https://www.kaggle.com/zeemeen/weighted-mean-running-10-sec-lb-0-509

sample_submission['air_store_id'] = sample_submission.id.map(lambda x: '_'.join(x.split('_')[:-1]))

sample_submission['calendar_date'] = sample_submission.id.map(lambda x: x.split('_')[2])
sample_submission.head(2) #air id and date is merged together
sample_submission.apply(lambda c: c.nunique()) 
air_store_info.apply(lambda x: x.nunique())
air_reserve.apply(lambda x: x.nunique())
set(air_reserve.air_store_id) < set(air_store_info.air_store_id) 
air_combine = pd.merge(air_reserve, air_store_info, on='air_store_id', how='outer') #joining reservation and store info data for air stores
air_combine.head(2)
hpg_combine = pd.merge(hpg_reserve, hpg_store_info, on='hpg_store_id', how='left') #joining reservation and store info data for hpg stores
hpg_combine.tail(2)
hpg_combine = pd.merge(hpg_combine, store_relation, on='hpg_store_id', how='right') #right join as we want data for only air stores
hpg_combine.head(2)
hpg_combine = hpg_combine.drop(['hpg_store_id'],axis =1) #don't require hpg_id now
hpg_combine.rename(columns={'hpg_genre_name': 'air_genre_name', 'hpg_area_name': 'air_area_name'}, inplace=True)#renaming column names to match up with the air_combine dataset
hpg_combine.tail(2)
air_combine.head(2)
air_combine.shape,hpg_combine.shape
air_combine = pd.concat([air_combine,hpg_combine],axis = 0) #combining data for air stores from both datasets
air_combine.tail(2)
air_combine.shape
air_combine['visit_date'] = pd.to_datetime(air_combine['visit_datetime'].dt.date)
air_combine['reserve_date'] = pd.to_datetime(air_combine['reserve_datetime'].dt.date)
air_combine.head(2)
air_combine = pd.merge(air_combine, date_info, left_on='visit_date',right_on='calendar_date', how='left') #joining on visit_date column
air_combine.head(1)
air_combine = air_combine.drop(['visit_datetime','reserve_datetime'],axis = 1) #dropping unnecessary columns
air_combine.head(2)
air_combine.isnull().sum() #null values
air_combine = air_combine.fillna({'reserve_visitors':0,'visit_date':pd.to_datetime('01/01/2099'),

                                  'reserve_date':pd.to_datetime('01/01/2099'),'calendar_date':pd.to_datetime('01/01/2099'),

                                  'day_of_week':'unknown','holiday_flg':-99,'latitude':-99,'longitude':-99,'air_genre_name':'unknown',

                                 'air_area_name':'unknown'})
air_combine.isnull().sum() #no null values
air_combine.dtypes
air_combine['holiday_flg'] = air_combine['holiday_flg'].astype('int8')



air_combine['day_of_week'] = air_combine['day_of_week'].astype('category')



air_combine['air_genre_name'] = air_combine['air_genre_name'].astype('category')



air_combine['air_area_name'] = air_combine['air_area_name'].astype('category')



air_combine['air_store_id'] = air_combine['air_store_id'].astype('category')



air_combine['reserve_visitors'] = air_combine['reserve_visitors'].astype('int8')
air_combine.dtypes
air_combine.head()
air_visit.head()
air_combine.to_csv('air_combine.csv',index = False)
# air_combine.to_feather('air_combine_raw') 
# df_combine = pd.read_feather('air_combine_raw') #to read from feather format
# df_combine.head()