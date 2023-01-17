!pip install pytrends
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import json



import pytrends

from pytrends.request import TrendReq

from pytrends import dailydata



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
path_dataset = '/kaggle/input/google-trends-keywords/'
with open(os.path.join(path_dataset, 'json_topic_edu_global.txt')) as json_data:

    dict_data = json.load(json_data)

    

df_topic_edu_global = pd.DataFrame(dict_data['default']['rankedList'][0]['rankedKeyword'])

df_topic_edu_global = pd.concat([df_topic_edu_global, df_topic_edu_global['topic'].apply(pd.Series)], axis=1)



with open(os.path.join(path_dataset, 'json_topic_work_global.txt')) as json_data:

    dict_data = json.load(json_data)

    

df_topic_work_global = pd.DataFrame(dict_data['default']['rankedList'][0]['rankedKeyword'])

df_topic_work_global = pd.concat([df_topic_work_global, df_topic_work_global['topic'].apply(pd.Series)], axis=1)



df_topic_global = pd.concat([df_topic_edu_global, df_topic_work_global], axis=0, ignore_index=True)

df_topic_global
kw_list = df_topic_global['title'].to_list()

kw_encode_list = df_topic_global['mid'].to_list()

geo_list = ['US', 'GB', 'IE', 'CA']
df_global_topic=pd.read_pickle(os.path.join(path_dataset, 'df_global_topic.pickle'))

df_global_topic
# Download daily data

for keyword, keyword_encode in zip(kw_list, kw_encode_list):

    for geo_id in geo_list:        

        if df_global_topic[(df_global_topic['keyword_encode']==keyword_encode)&(df_global_topic['geo_id']==geo_id)].shape[0]==0:

            df_daily = dailydata.get_daily_data(keyword_encode, 2016, 1, 2018, 12, geo = geo_id)

            df_daily['keyword'] = keyword

            df_daily['keyword_encode'] = keyword_encode

            df_daily['geo_id'] = geo_id

            df_daily.columns = df_daily.columns.str.replace(keyword_encode, 'trend')



            df_global_topic = pd.concat([df_global_topic, df_daily], axis=0)
df_global_topic.pivot_table(index='keyword', columns='geo_id', values='trend', aggfunc='count')
# Merge and export the data

df_global_topic = df_global_topic.reset_index().drop_duplicates(subset=['date','geo_id','keyword_encode']).set_index('date')



df_global_topic.to_pickle('df_global_topic.pickle')

df_global_topic