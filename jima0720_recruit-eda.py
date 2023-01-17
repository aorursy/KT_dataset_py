# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd

data_path = '../input/'
air_store_info = pd.read_csv(data_path + 'air_store_info.csv')
hpg_store_info = pd.read_csv(data_path + 'hpg_store_info.csv')
sample_submission = pd.read_csv(data_path + 'sample_submission.csv')
date_info = pd.read_csv(data_path + 'date_info.csv')
hpg_reserve = pd.read_csv(data_path + 'hpg_reserve.csv')
air_visit_data = pd.read_csv(data_path + 'air_visit_data.csv')
air_reserve = pd.read_csv(data_path + 'air_reserve.csv')
store_id_relation = pd.read_csv(data_path + 'store_id_relation.csv')

from IPython.core.display import display


display(air_store_info.head(3))

display(air_store_info.describe())

display(hpg_store_info.head(3))
display(hpg_store_info.describe())
display(sample_submission.head(3))
display(date_info.head(5))

# display(date_info.describe())
display(hpg_reserve.head(5))
# display(hpg_reserve.describe())
air_visit_data.head(3)
air_reserve.head(3)
store_id_relation.head(3)
import gc

# 平日/休日の平均客数
tmp = pd.merge(air_visit_data, date_info, left_on='visit_date', right_on='calendar_date')
tmp.groupby('holiday_flg').mean()

tmp2 = sample_submission

# sample_submissionファイルから訪問日時列を作成
# air_00a91d42b08b08d9_2017-04-23
tmp2['visit_date'] = sample_submission['id'].str[-10:]
display(tmp2.head(3))

# カレンダーTBLとjoinして平日/休日を見分けるようにしてみる
tmp3 = pd.merge(tmp2, date_info, left_on='visit_date', right_on='calendar_date')
display(tmp3.head(3))

# 平日/休日ごとの平均客数を埋め込む
o_avg = 21
h_avg = 24
tmp3.loc[ tmp3['holiday_flg'] == 1, 'visitors' ] = h_avg
tmp3.loc[ tmp3['holiday_flg'] == 0, 'visitors' ] = o_avg

# 結果確認
display(tmp3.loc[ tmp3['holiday_flg'] == 1].head(3))
display(tmp3.loc[ tmp3['holiday_flg'] == 0].head(3))

# 必要なカラムのみに絞る
submit_df = tmp3[['id', 'visitors']]
display(submit_df.head(3))


# submissionファイル作成
from datetime import datetime, timedelta, timezone
JST = timezone(timedelta(hours=+9), 'JST')
ts = datetime.now(JST).strftime('%y%m%d%H%M')

submit_df.to_csv(('submit_'+ts+'.csv'),index=False)
