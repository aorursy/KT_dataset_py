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
print(dirname+'/sample_submission.csv')

test_df = pd.read_csv(dirname+'/sample_submission.csv')

test_df.head(20)

test_df['store_id'] = test_df['id'].str[:20]

test_df['visit_date'] = test_df['id'].str[21:]



test_df.drop(['visitors'],axis=1,inplace=True)



test_df['visit_date'] = pd.to_datetime(test_df['visit_date'])



test_df.info()



test_df.head()
air_data = pd.read_csv(dirname+'/air_visit_data.csv',parse_dates=['visit_date'])



air_data.shape



air_data.head()
check_store_sample = air_data[air_data['air_store_id']=='air_00a91d42b08b08d9']

check_store_sample.describe()



check_store_sample.visit_date.describe()
air_data['dow']=air_data['visit_date'].dt.dayofweek

train=air_data[air_data['visit_date']>'2017-01-28'].reset_index()

train['dow']=train['visit_date'].dt.dayofweek

test_df['dow']=test_df['visit_date'].dt.dayofweek



test_df.head()

train.head()
aggregation={'visitors':'median'}



agg_data=train.groupby(['air_store_id','dow']).agg(aggregation).reset_index()



agg_data.colums=['air_store_id','dow','visitors']

agg_data['visitors']=agg_data['visitors']



agg_data.head(12)
marged=pd.merge(test_df,agg_data,how='left',left_on=['store_id','dow'],right_on=['air_store_id','dow'])



marged.head()

#marged.index.values

final = marged[['id','visitors']]



final.head()
def missing_values_table(df):

    mis_val = df.isnull().sum()

    mis_val_percent=100*df.isnull().sum()/len(df)

    mis_val_table=pd.concat([mis_val,mis_val_percent],axis=1)

    mis_val_table_ren_columns=mis_val_table.rename(

    columns={0:'MisiingValues',1:'% of Total Values'})

    return mis_val_table_ren_columns

missing_values_table(final)
final.fillna(0,inplace=True)



missing_values_table(final)
import glob

import re

import os

# 全てのCSVを一気に読み込む

# glob.glob('')に適切なファイルのパスを指定してください

dfs = { re.search('/([^/\.]*)\.csv', fn).group(1):pd.read_csv(fn) for fn in glob.glob(dirname+'/*.csv')}

for k, v in dfs.items(): locals()[k] = v

# 読み込んだファイルを確認

print('data frames read:{}'.format(list(dfs.keys())))

#data frames read:['air_store_info', 'date_info', 'store_id_relation', 'hpg_reserve', 'air_reserve', 'air_visit_data', 'sample_submission', 'hpg_store_info']

date_info[date_info['holiday_flg']==1].head(10)
weekend_hdays=date_info.apply((lambda x:(x.day_of_week=='Sunday' or x.day_of_week=='Saturday')and x.holiday_flg==1),axis=1)



date_info.loc[weekend_hdays, 'holiday_flg']=0
date_info['weight']=(date_info.index+1)/len(date_info)



date_info.head()

date_info.tail()
visit_data=air_visit_data.merge(date_info,left_on='visit_date',right_on='calendar_date',how='left')

visit_data.drop('calendar_date',axis=1,inplace=True)

visit_data['visitors']=visit_data.visitors.map(pd.np.log1p)



visit_data.head(10)
wmean=lambda x:((x.weight * x.visitors).sum()/x.weight.sum())



visitors=visit_data.groupby(

['air_store_id','day_of_week','holiday_flg']).apply(wmean).reset_index()

visitors.rename(columns={0:'visitors'},inplace=True)



visitors.head(10)
sample_submission['air_store_id']=sample_submission.id.map(

    lambda x:'_'.join(x.split('_')[:-1]))

sample_submission['calendar_date']=sample_submission.id.map(lambda x: x.split('_')[2])



sample_submission.drop('visitors',axis=1,inplace=True)

sample_submission=sample_submission.merge(date_info, on='calendar_date',how='left')

sample_submission=sample_submission.merge(

visitors, on=['air_store_id','day_of_week','holiday_flg'],how='left')

sample_submission.head(10)
missing_values_table(sample_submission)
missings=sample_submission.visitors.isnull()

sample_submission.loc[missings,'visitors']=sample_submission[missings].merge(

visitors[visitors.holiday_flg==0],on=('air_store_id','day_of_week'), how='left')['visitors_y'].values



missing_values_table(sample_submission)
missings=sample_submission.visitors.isnull()

sample_submission.loc[missings,'visitors']=sample_submission[missings].merge(

visitors[['air_store_id','visitors']].groupby('air_store_id').mean().reset_index(),on='air_store_id',how='left')['visitors_y'].values



missing_values_table(sample_submission)
sample_submission['visitors']=sample_submission.visitors.map(pd.np.expm1)



sample_submission=sample_submission[['id','visitors']]

final['visitors'][final['visitors']==0]=sample_submission['visitors'][final['visitors']==0]

sub_file=final.copy()



sub_file.head()



sub_file['visitors']=np.mean([final['visitors'],sample_submission['visitors']],axis=0)

sub_file.to_csv('sub_math_mean_1.csv',index=False)



sub_file['visitors']=(final['visitors']*sample_submission['visitors'])**(1/2)

sub_file.to_csv('sub_geo_mean_1.csv',index=False)



sub_file['visitors']=2/(1/final['visitors']+1/sample_submission['visitors'])

sub_file.to_csv('sub_hrm_mean_1.csv',index=False)
