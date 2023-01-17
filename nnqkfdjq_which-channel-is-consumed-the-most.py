import pandas as pd
import numpy as np
import holoviews as hv
from datetime import datetime, timedelta
import json
import sqlite3
from sqlalchemy import create_engine
from urllib.parse   import quote
from urllib.request import urlopen
import time
import matplotlib.pyplot as plt
import re
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import wordnet
hv.extension('bokeh')

fm = pd.read_csv("../input/statistics-observation-of-random-youtube-video/count_observation_upload.csv")
fm2 = pd.read_csv("../input/statistics-observation-of-random-youtube-video/video_characteristics_upload.csv")
fm2 = fm2.drop('Unnamed: 0', axis = 1)
fm = fm.drop('Unnamed: 0', axis = 1)
fm = fm.set_index('index')
datetime_tran2 = lambda x : datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
fm.loc[:,['commentCount', 'dislikeCount', 'favoriteCount', 'likeCount', 'viewCount']] =  fm.loc[:,['commentCount', 'dislikeCount', 'favoriteCount', 'likeCount', 'viewCount']].astype(np.float)
fm['Time'] = fm['Time'].map(datetime_tran2)
videoId_list = list(fm.videoId.unique())
vi_cat_dict = fm2.loc[:,['videoId','categoryId']].set_index('videoId').to_dict()['categoryId']
fm['categoryId'] = fm['videoId'].map(vi_cat_dict)
get_hour = lambda x : x.hour 
fm['Hour'] = fm['Time'].map(get_hour)
vi_tit_dict = fm2.loc[:,['videoId','title']].set_index('videoId').to_dict()['title']
fm['title'] = fm['videoId'].map(vi_tit_dict)
with open('../input/youtube-new/US_category_id.json') as fb:
    fb = fb.read()
    categoryId_to_name = json.loads(fb)
categoryId_to_name2 = {}
for item in categoryId_to_name['items']:
    categoryId_to_name2[np.float(item['id'])] = item['snippet']['title'] 
fm['categoryId'] = fm['categoryId'].map(categoryId_to_name2)
%opts Bars [stack_index=1 xrotation=0 width=800 height=500 show_legend=False tools=['hover']]
%opts Bars (color=Cycle('Category20'))
def top10(x):
    return x.sort_values(by = 'viewCount_diff', ascending = False)[0:10]
kingking = re.compile(r"official|music|lyric|ft|mv")
def music_goaway(title):   
    if type(title) != str: #title이 str 아니면 빼고 
        return False
    title = title.lower()
    if len(re.findall(kingking, title)) > 0:
        return False
    else:
        return True
vi_ch_dict = fm2.loc[:,['videoId','channelTitle']].set_index('videoId').to_dict()['channelTitle'] #fm2로 카테고리 아이디랑 비디오아이디 매칭 딕
fm['channelTitle'] = fm['videoId'].map(vi_ch_dict)
channel_fm = fm.loc[:,['Hour','channelTitle','viewCount_diff']]
channel_sort = channel_fm.groupby(['Hour','channelTitle']).mean() #시간대 별 채널로 모임. 
channel_sort2 = channel_sort.reset_index()
channel_sort2 = channel_sort2.groupby('Hour').apply(top10)
channel_sort2 = channel_sort2.drop('Hour', axis = 1).reset_index()
key_dimensions2   = [('Hour', 'Hour'), ('channelTitle', 'channelTitle')]
value_dimensions2 = [('viewCount_diff', 'viewCount_diff')]
macro3 = hv.Table(channel_sort2, key_dimensions2, value_dimensions2)
macro3.to.bars(['Hour','channelTitle'], 'viewCount_diff', [], label = "what??? ")
target_fm = fm[fm['categoryId'] != 'Music'] 
target_fm2 = target_fm[target_fm['title'].map(music_goaway)]
channel_fm1 = target_fm2.loc[:,['Hour','channelTitle','viewCount_diff']]
channel_sort1 = channel_fm1.groupby(['Hour','channelTitle']).mean() #시간대 별 채널로 모임. 
channel_sort21 = channel_sort1.reset_index()
channel_sort21 = channel_sort21.groupby('Hour').apply(top10)
channel_sort21 = channel_sort21.drop('Hour', axis = 1).reset_index()
key_dimensions21   = [('Hour', 'Hour'), ('channelTitle', 'channelTitle')]
value_dimensions21 = [('viewCount_diff', 'viewCount_diff')]
macro31 = hv.Table(channel_sort21, key_dimensions21, value_dimensions21)
macro31.to.bars(['Hour','channelTitle'], 'viewCount_diff', [], label = "what??? ")
channel_sort = fm.loc[:,['Time','channelTitle','viewCount_diff']]
channel_sort = channel_sort.groupby(['Time','channelTitle']).mean()
channel_sort2 = channel_sort.reset_index()
channel_sort2 = channel_sort2.groupby('Time').apply(top10)
channel_sort2 = channel_sort2.drop('Time', axis = 1).reset_index()
key_dimensions2   = [('Time', 'Time'), ('channelTitle', 'channelTitle')]
value_dimensions2 = [('viewCount_diff', 'viewCount_diff')]
macro3 = hv.Table(channel_sort2, key_dimensions2, value_dimensions2)
macro3.to.bars(['Time','channelTitle'], 'viewCount_diff', [], label = "channels view count ranking each Time") 
target_fm = fm[fm['categoryId'] != 'Music'] 
target_fm2 = target_fm[target_fm['title'].map(music_goaway)]
channel_sort = target_fm2.loc[:,['Time','channelTitle','viewCount_diff']]
channel_sort = channel_sort.groupby(['Time','channelTitle']).mean()
channel_sort2 = channel_sort.reset_index()
channel_sort2 = channel_sort2.groupby('Time').apply(top10)
channel_sort2 = channel_sort2.drop('Time', axis = 1).reset_index()
key_dimensions2   = [('Time', 'Time'), ('channelTitle', 'channelTitle')]
value_dimensions2 = [('viewCount_diff', 'viewCount_diff')]
macro3 = hv.Table(channel_sort2, key_dimensions2, value_dimensions2)
macro3.to.bars(['Time','channelTitle'], 'viewCount_diff', [], label ="channels view count ranking each Time, excluded music") 

target_fm = fm[fm['categoryId'] != 'Music'] 
title_sort = target_fm.loc[:,['Time','title','viewCount_diff']]
title_sort = title_sort[title_sort['title'].map(music_goaway)]
title_sort = title_sort.groupby(['Time','title']).sum().reset_index() 
title_sort = title_sort.groupby('Time').apply(top10)
title_sort = title_sort.drop('Time', axis = 1).reset_index()
key_dimensions31   = [('Time', 'Time'), ('title', 'title')]
value_dimensions31 = [('viewCount_diff', 'viewCount_diff')]
macro41 = hv.Table(title_sort, key_dimensions31, value_dimensions31)
macro41.to.bars(['Time','title'], 'viewCount_diff', []) 




