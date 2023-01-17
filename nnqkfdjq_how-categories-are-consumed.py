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
with open('../input/youtube-new/US_category_id.json') as fb:
    fb = fb.read()
    categoryId_to_name = json.loads(fb)
categoryId_to_name2 = {}
for item in categoryId_to_name['items']:
    categoryId_to_name2[np.float(item['id'])] = item['snippet']['title'] 
fm['categoryId'] = fm['categoryId'].map(categoryId_to_name2)
get_hour = lambda x : x.hour
categoryId_sort = fm.groupby(['categoryId',fm.Time.map(get_hour)]).mean()
category_list = list(fm['categoryId'].unique()) 
category_list.remove(np.nan)
fm['Hour'] = fm['Time'].map(get_hour)
for cat in category_list: 
    for Time in range(24):
        std = np.std(fm.loc[(fm['categoryId'] == cat)&(fm['Hour'] == Time),'viewCount_diff'])
        categoryId_sort.loc[(cat,Time),'std_of_the_Hour'] = std
   # print("{0} is done".format(cat))
%opts Curve [height=400 width=1000, tools=['hover'], toolbar='above']
third_curve_data = hv.Dataset(fm[fm['videoId'] == '9jI-z9QN6g8'].loc[:,['Time', 'viewCount_diff']] , vdims = 'viewCount_diff', kdims = 'Time')
hv.Curve(third_curve_data, label = "viewCount_difference of every hour [video title : Te Bote Remix - Casper, Nio García, Darell, Nicky Jam, Bad Bunny, Ozuna | Video Oficial]")
fig, ax = plt.subplots(9, 2, figsize=(13,70))
org = 0
color1 = plt.cm.viridis(.9)
for num, cat in enumerate(category_list):
    m = categoryId_sort.loc[cat,['viewCount_diff','std_of_the_Hour']].reset_index()
    if num%2 == 0:
        ax[num-org,0].bar(np.arange(len(m)),m['viewCount_diff'], align='center', alpha=0.5, label = 'view Mean of each Hour')
        ax[num-org,0].set_xticks(np.arange(len(m)))
        ax[num-org,0].set_xticklabels(m['Time'])
        ax[num-org,0].set_xlabel("Hour")
        ax[num-org,0].set_ylabel("view Mean of each Hour")
        ax[num-org,0].set_title("{0}".format(cat))
        ax_twin = ax[num-org,0].twinx()
        ax_twin.bar(np.arange(len(m)),m['std_of_the_Hour'], color=color1, align='center', alpha=0.5, label = "Std of each Hour")
        ax_twin.set_ylabel("Std of each Hour")
        ax_twin.set_ylim(0, m['std_of_the_Hour'].max()*2.5)
        lines, labels = ax[num-org,0].get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        ax_twin.legend(lines + lines2, labels + labels2, loc=0)
        org +=1
    else:
        ax[num-org,1].bar(np.arange(len(m)),m['viewCount_diff'], align='center', alpha=0.5, label = 'view Mean of each Hour')
        ax[num-org,1].set_xticks(np.arange(len(m)))
        ax[num-org,1].set_xticklabels(m['Time'])
        ax[num-org,1].set_xlabel("Hour")
        ax[num-org,1].set_ylabel("view Mean of each Hour")
        ax[num-org,1].set_title("{0}".format(cat))
        ax_twin = ax[num-org,1].twinx()
        ax_twin.bar(np.arange(len(m)),m['std_of_the_Hour'], color=color1, align='center', alpha=0.5, label = "Std of each Hour")
        ax_twin.set_ylabel("Std of each Hour")
        ax_twin.set_ylim(0, m['std_of_the_Hour'].max()*2.5)
        lines, labels = ax[num-org,1].get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        ax_twin.legend(lines + lines2, labels + labels2, loc=0)
   # print("{0} done {1}".format(num, cat))
fig.suptitle("View mean and std of each hour", fontsize=20)
%matplotlib inline
fig
fm3 = fm.loc[:,['categoryId','viewCount_diff','Hour']]
fm4 = fm3.groupby(['Hour','categoryId']).sum()
fm4 = fm4.reset_index()
key_dimensions   = [('Hour', 'Hour'), ('categoryId', 'categoryId')]
value_dimensions = [('viewCount_diff', 'viewCount_diff')]
macro = hv.Table(fm4, key_dimensions, value_dimensions)
%opts Bars [stack_index=1 xrotation=0 width=800 height=500 show_legend=False tools=['hover']]
%opts Bars (color=Cycle('Category20'))
macro.to.bars(['Hour','categoryId'], 'viewCount_diff', [], label = "View sum comparison of every category") #
fm5 = fm4[fm4['categoryId'] != 'Music']
macro2 = hv.Table(fm5, key_dimensions, value_dimensions)
macro2.to.bars(['Hour','categoryId'], 'viewCount_diff', [], label = "View sum comparison of every category - exclude Music category")
def top_sort(x):
    return x.sort_values(by = 'viewCount_diff', ascending = False)
category_Time = fm.groupby(['Time','categoryId'])['viewCount_diff'].mean().reset_index()
category_Time = category_Time.groupby('Time').apply(top_sort)
category_Time = category_Time.drop('Time',axis = 1).reset_index()
key_dimensions33   = [('Time', 'Time'), ('categoryId', 'categoryId')]
value_dimensions33 = [('viewCount_diff', 'viewCount_diff')]
macro43 = hv.Table(category_Time, key_dimensions33, value_dimensions33)
macro43.to.bars(['Time','categoryId'], 'viewCount_diff', []) 
target_fm = fm[fm['categoryId'] !='Music']
category_Time = target_fm.groupby(['Time','categoryId'])['viewCount_diff'].mean().reset_index()
category_Time = category_Time.groupby('Time').apply(top_sort)
category_Time = category_Time.drop('Time',axis = 1).reset_index()
key_dimensions33   = [('Time', 'Time'), ('categoryId', 'categoryId')]
value_dimensions33 = [('viewCount_diff', 'viewCount_diff')]
macro43 = hv.Table(category_Time, key_dimensions33, value_dimensions33)
macro43.to.bars(['Time','categoryId'], 'viewCount_diff', []) 