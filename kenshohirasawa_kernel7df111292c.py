# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
air_visit_data = pd.read_csv('/kaggle/input/recruit-restaurant-visitor-forecasting/air_visit_data.csv.zip')
air_visit_data.head()
date_info = pd.read_csv('/kaggle/input/recruit-restaurant-visitor-forecasting/date_info.csv.zip')
date_info.head()
sample_submission = pd.read_csv('/kaggle/input/recruit-restaurant-visitor-forecasting/sample_submission.csv.zip')
sample_submission.head()
# join しやすい形式にサンプル提出データを変換
base_submission = pd.DataFrame({
    'air_store_id': sample_submission['id'].apply(lambda x : x[:-11]),
    'visit_date': sample_submission['id'].apply(lambda x : x[-10:]),
    'visitors': sample_submission['visitors']
})
base_submission.head()
# 訪問数と日付情報をJOIN
air_visit_with_date = pd.merge(air_visit_data, date_info, left_on='visit_date', right_on='calendar_date')
air_visit_with_date.head()
# 店・曜日ごとに平均を出す
air_visit_wod = air_visit_with_date.groupby(['air_store_id', 'day_of_week'])['visitors'].mean()
air_visit_wod.head()
# 店・曜日ごとの平均で提出データを作る
wod_mean_submission = pd.merge(base_submission, date_info, left_on='visit_date', right_on='calendar_date')
wod_mean_submission = pd.merge(wod_mean_submission, air_visit_wod, how='left', on=['air_store_id', 'day_of_week'])
wod_mean_submission = wod_mean_submission.fillna(value=0.0) # 本当は店の平均で埋めたほうがいいかも
wod_mean_submission = pd.DataFrame({
    'id': wod_mean_submission['air_store_id'] + '_' + wod_mean_submission['visit_date'],
    'visitors': wod_mean_submission['visitors_y']
})
wod_mean_submission.head()
# 保存
wod_mean_submission.to_csv('/kaggle/working/wod_mean_submission.csv.zip', index=False, compression='zip')
# 祝日の場合の平均
air_visit_holiday = air_visit_with_date.groupby(['air_store_id', 'holiday_flg'])['visitors'].mean()
air_visit_holiday.head()
# 平日の曜日ごと平均
air_visit_no_holiday = air_visit_with_date[air_visit_with_date['holiday_flg'] == 0]
air_visit_wod_no_holiday = air_visit_no_holiday.groupby(['air_store_id', 'day_of_week'])['visitors'].mean()
air_visit_wod_no_holiday.head()
# 店の平均
air_visit_mean = air_visit_data.groupby('air_store_id')['visitors'].mean()
air_visit_mean.head()
base_submission_with_date = pd.merge(base_submission, date_info, left_on='visit_date', right_on='calendar_date')
base_submission_holiday = base_submission_with_date[base_submission_with_date['holiday_flg'] == 1]
base_submission_no_holiday = base_submission_with_date[base_submission_with_date['holiday_flg'] == 0]
holiday_df = pd.merge(base_submission_holiday, air_visit_holiday, how='left', on=['air_store_id', 'holiday_flg'])
no_holiday_df = pd.merge(base_submission_no_holiday, air_visit_wod_no_holiday, how='left', on=['air_store_id', 'day_of_week'])
tmp_df = pd.concat([holiday_df, no_holiday_df])
tmp_df = pd.merge(tmp_df, air_visit_mean, how='left', on='air_store_id')
tmp_df = tmp_df.fillna(tmp_df["visitors"])
tmp_df = tmp_df.fillna(value=0.0)
holiday_submission = pd.DataFrame({
    'id': tmp_df['air_store_id'] + '_' + tmp_df['visit_date'],
    'visitors': tmp_df['visitors_y']
})
# 保存
holiday_submission.to_csv('/kaggle/working/holiday_submission.csv.zip', index=False, compression='zip')
