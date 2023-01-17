# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import pandas_profiling
import string
from IPython.display import display
import plotly.graph_objs as go
import plotly.express as px
import plotly
plotly.offline.init_notebook_mode(connected=True)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dc = pd.read_csv('/kaggle/input/fivethirtyeight-comic-characters-dataset/dc-wikia-data.csv')
marvel = pd.read_csv('/kaggle/input/fivethirtyeight-comic-characters-dataset/marvel-wikia-data.csv')
dc = dc.rename(columns={'page_id':'Page_ID',
                       'name': 'Name',
                       'urlslug':'Urlslug',
                       'ID':'ID',
                       'ALIGN': 'Align',
                       'EYE':'Eye',
                       'HAIR':'Hair',
                       'SEX':'Sex',
                       'GSM':'Gsm',
                       'ALIVE': 'Alive',
                       'APPEARANCES':'Appearances',
                       'FIRST APPEARANCE': 'First Appearance',
                       'YEAR': 'Year'}
)
marvel = marvel.rename(columns={'page_id':'Page_ID',
                       'name': 'Name',
                       'urlslug':'Urlslug',
                       'ID':'ID',
                       'ALIGN': 'Align',
                       'EYE':'Eye',
                       'HAIR':'Hair',
                       'SEX':'Sex',
                       'GSM':'Gsm',
                       'ALIVE': 'Alive',
                       'APPEARANCES':'Appearances',
                       'FIRST APPEARANCE': 'First Appearance',
                       'Year': 'Year'}
)
alive_m = marvel.Alive.value_counts()
alive_dc = dc.Alive.value_counts()
sex_m = marvel.Sex.value_counts()
sex_dc = dc.Sex.value_counts()
m_identity = marvel.ID.value_counts().sort_values(ascending=True)
data = [go.Pie(
        labels = m_identity.index,
        values = m_identity.values,
        hoverinfo = 'label+value'
)]

plotly.offline.iplot(data, filename='active_category')
dc_identity = dc.ID.value_counts().sort_values(ascending=True)
data = [go.Pie(
        labels = dc_identity.index,
        values = dc_identity.values,
        hoverinfo = 'label+value'
)]

plotly.offline.iplot(data, filename='active_category')
m_sex = sex_m.sort_values(ascending=False)
data = [go.Pie(labels = m_sex.index,
               values = m_sex.values,
               hoverinfo = 'label+value'
)]

plotly.offline.iplot(data)
dc_sex = sex_dc.sort_values(ascending=False)
data = [go.Pie(labels = dc_sex.index,
               values = dc_sex.values,
               hoverinfo = 'label+value'
)]

plotly.offline.iplot(data)
alive_m = marvel.Alive.value_counts()
alive_dc = dc.Alive.value_counts()
alive_m = alive_m.sort_values(ascending=False)
data = [go.Pie(labels = alive_m.index,
               values = alive_m.values,
               hoverinfo = 'label+value'
)]

plotly.offline.iplot(data)
alive_dc = alive_dc.sort_values(ascending=False)
data = [go.Pie(labels = alive_dc.index,
               values = alive_dc.values,
               hoverinfo = 'label+value'
)]

plotly.offline.iplot(data)
top_10_appearances_m = marvel.sort_values('Appearances', ascending=False).head(10)
top_10_appearances_dc = dc.sort_values('Appearances', ascending=False).head(10)
align_count = top_10_appearances_m.groupby('Name')['Align'].sum()
fig,(ax1) = plt.subplots(figsize=(15,7))
sns.countplot(x=align_count.index, data = align_count, ax=ax1)
plt.ylabel('Number of Character')
plt.show()
new_heroes_m = marvel[(marvel['Year'] >= 2000)]
new_heroes_dc = dc[(dc['Year'] >= 2000)]
sns.set_style('whitegrid')
fig,(ax1) = plt.subplots(figsize=(20,11))
new_heroes_align_m = new_heroes_m.groupby('Name')['Align'].sum()
plt.title('Number of different Characters types in Marvel universe')
sns.countplot(x=new_heroes_align_m.index, data=new_heroes_align_m, ax=ax1)
plt.show()
new_heroes_align_dc = new_heroes_dc.groupby('Name')['Align'].sum()
fig,(ax1) = plt.subplots(figsize=(20,11))
plt.title('Number of different Characters types in DC universe')
sns.countplot(x=new_heroes_align_dc.index, data=new_heroes_align_dc, ax=ax1)
plt.show()
sns.set_style('darkgrid')
fig,(ax1) = plt.subplots(figsize=(20,11))
sns.kdeplot(data=marvel['Year'], label='Marvel', shade=True)
sns.kdeplot(data=dc['Year'], label='DC', shade=True, ax=ax1)
plt.title('Distribution of appearance of heroes in comic in years')
plt.show()
top_10_appearances_m = top_10_appearances_m.sort_values('Appearances', ascending=False)
data = [go.Pie(labels = top_10_appearances_m['Name'],
               values = top_10_appearances_m['Appearances'],
               hoverinfo = 'label+value'
)]

plotly.offline.iplot(data)
top_10_appearances_dc = top_10_appearances_dc.sort_values('Appearances', ascending=False)
data = [go.Pie(labels = top_10_appearances_dc['Name'],
               values = top_10_appearances_dc['Appearances'],
               hoverinfo = 'label+value'
)]

plotly.offline.iplot(data)
dc_marvel = pd.concat([dc, marvel])
top_10_appearances_dc_marvel = dc_marvel.sort_values('Appearances', ascending=False).head(10)
top_10_appearances_dc_marvel = top_10_appearances_dc_marvel.sort_values('Appearances', ascending=False)
data = [go.Pie(labels = top_10_appearances_dc_marvel['Name'],
               values = top_10_appearances_dc_marvel['Appearances'],
               hoverinfo = 'label+value'
)]

plotly.offline.iplot(data)