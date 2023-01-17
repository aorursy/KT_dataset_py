#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'data:image/jpeg;base64,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',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.offline as py

import plotly.graph_objs as go

import plotly.offline as py

import plotly.express as px



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/bots-ua-parsed/bots_full.csv', encoding='ISO-8859-2')

df.head()
sns.countplot(df['os'],linewidth=3,palette="Set2",edgecolor='black')

plt.xticks(rotation=45)

plt.yticks(rotation=45)

plt.show()
na_percent = (df.isnull().sum()/len(df))[(df.isnull().sum()/len(df))>0].sort_values(ascending=False)



missing_data = pd.DataFrame({'Missing Percentage':na_percent*100})

missing_data
na = (df.isnull().sum() / len(df)) * 100

na = na.drop(na[na == 0].index).sort_values(ascending=False)



f, ax = plt.subplots(figsize=(12,8))

sns.barplot(x=na.index, y=na)

plt.xticks(rotation='90')

plt.xlabel('Features', fontsize=15)

plt.title('Percentage Missing', fontsize=15)
for col in ('device_type', 'sec_lvl_domn', 'page_vw_ts', 'user_os'):

    df[col] = df[col].fillna('None')
for col in ('city', 'st', 'user_browser', 'is_bot', 'user_agent', 'page_url'):

    df[col] = df[col].fillna(df[col].mode()[0])
bots_full=df.sort_values(by='operating_sys', ascending=False)

bots_full.head(10)

cols=['sec_lvl_domn','page_url']

bots_full=bots_full[cols].head(10)

bots_full=bots_full.set_index('sec_lvl_domn')
from IPython.display import Image, HTML

def path_to_image_html(path):

    '''

     This function essentially convert the image url to 

     '<img src="'+ path + '"/>' format. And one can put any

     formatting adjustments to control the height, aspect ratio, size etc.

     within as in the below example. 

    '''



    return '<img src="'+ path + '"width="60" height="60""/>'



HTML(bots_full.to_html(escape=False ,formatters=dict(img=path_to_image_html),justify='center'))
df=df.sort_values(by='VIEWS', ascending=False)

bot=df.head(10)

colors = ['rgb(239, 243, 255)', 'rgb(189, 215, 231)', 'rgb(107, 174, 214)',

          'rgb(49, 130, 189)', 'rgb(8, 81, 156)',]

fig = go.Figure(data=[go.Table(header=dict(values=['sec_lvl_domn','VIEWS'],fill_color='black',font=dict(color='white', size=12)

),

                 cells=dict(values=[list(bot['sec_lvl_domn']),list(bot['VIEWS'])],fill_color='rgb(107, 174, 214)',font=dict(color='black', size=12)))

                     ])

fig.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcT9nr4HKUtvinWMAKUYu9GtH2jT-DZx-0ArDdJmiMRIUAuu8EvQ&usqp=CAU',width=400,height=400)