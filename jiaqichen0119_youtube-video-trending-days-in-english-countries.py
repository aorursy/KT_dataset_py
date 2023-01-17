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
import time

import datetime

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.colors as colors

import plotly

%matplotlib inline
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.figure_factory as ff

import plotly.tools as tls
us_videos = pd.read_csv('/kaggle/input/youtube-new/USvideos.csv')

ca_videos = pd.read_csv('/kaggle/input/youtube-new/CAvideos.csv')

gb_videos = pd.read_csv('/kaggle/input/youtube-new/GBvideos.csv')
us_videos['trending_date'] = pd.to_datetime(us_videos['trending_date'],format='%y.%d.%m')

ca_videos['trending_date'] = pd.to_datetime(ca_videos['trending_date'],format='%y.%d.%m')

gb_videos['trending_date'] = pd.to_datetime(gb_videos['trending_date'],format='%y.%d.%m')
us_videos['trending_date'].sort_values
x_d = list(us_videos.trending_date.value_counts().index.values)

y_u = list(us_videos.trending_date.value_counts().values)

y_c = list(ca_videos.trending_date.value_counts().values)

y_g = list(gb_videos.trending_date.value_counts().values)
import matplotlib.dates as mdate
plt.figure(figsize = (30,4))

plt.bar(x_d,y_u,width =0.3,label = 'America',align='center',alpha = 0.5,fc = 'g')

plt.bar(x_d,y_c,width =0.3,label = 'Cananda',align='edge',alpha = 0.5,fc = 'r')

plt.bar(x_d,y_g,width =-0.3,label = 'Unit Kingdom',align='edge',alpha = 0.5,fc = 'y')

ax1 = plt.gca()

ax1.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))

plt.xticks(pd.date_range('2017-11-15','2018-06-14',freq='d'),rotation =90,ha='left',fontsize =10)

plt.ylabel('number of trending videos',fontsize =10)

# plt.legend(loc='EastOutside')

plt.xlim('2017-11-14','2018-06-15')

plt.title('daily trending day distribution',fontsize = 15)

box = ax1.get_position()

ax1.set_position([box.x0, box.y0, box.width , box.height* 0.8])

ax1.legend(loc='center left', bbox_to_anchor=(0.2, 1.12),ncol=3)
en_videos = pd.concat([us_videos,ca_videos,gb_videos],axis = 0)

en_videos_cleaned = en_videos.drop_duplicates(['video_id','trending_date'],keep = 'first')
frame1 = en_videos_cleaned.drop_duplicates(['video_id'],keep = 'first').set_index('video_id')[['category_id','publish_time','tags','title','channel_title','description','thumbnail_link']]
frame2 = pd.DataFrame(en_videos_cleaned.groupby('video_id')['views','likes','dislikes','comment_count'].max())
frame3 = pd.DataFrame(en_videos_cleaned.groupby('video_id')['trending_date'].min())
frame4 = pd.DataFrame(en_videos_cleaned.groupby('video_id').count()['trending_date']).rename(columns={'trending_date':'trending_days'},inplace=False)
frame1.head()
frame5 = pd.merge(frame4,frame3,left_index=True, right_index=True)

frame6 = pd.merge(frame5,frame2,left_index=True, right_index=True)

en_videos_dataset = pd.merge(frame6,frame1,left_index=True, right_index=True).reset_index()
en_videos_dataset.head()
us_videos_categories = pd.read_json('/kaggle/input/youtube-new/US_category_id.json')

us_categories = {category['id']:category['snippet']['title'] for 

                category in us_videos_categories['items']}
us_categories
en_videos_dataset['category_name'] = en_videos_dataset['category_id'].apply(lambda x:us_categories.get(str(x)))
diff = en_videos_dataset[['video_id','trending_date']]

diff['publish_date'] = en_videos_dataset['publish_time'].apply(lambda x:x.split('T')[0])

diff['publish_date'] = pd.to_datetime(diff['publish_date'],format='%Y-%m-%d')

diff['trending_date'] = en_videos_dataset['trending_date']

diff['days_to_trend'] = diff['trending_date'] - diff['publish_date']

diff['days_to_trend'] = diff['days_to_trend'].dt.days

en_videos_dataset['days_to_trend'] = diff['days_to_trend']
en_videos_dataset.head()
import pandas_profiling as pp
en_videos_dataset.columns
pp.ProfileReport(en_videos_dataset[['trending_days','days_to_trend', 'views', 'likes',

       'dislikes', 'comment_count']])
en_videos_dataset.sort_values(by = ['trending_days','views'],ascending=False)[:10]
en_videos_dataset.sort_values(by = ['trending_days','views'],ascending=False)[:10].describe()