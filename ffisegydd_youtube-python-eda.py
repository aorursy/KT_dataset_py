# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import json
from pathlib import Path

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12, 8)
from matplotlib import pyplot as plt

from gensim.models.word2vec import Word2Vec

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
def load_category(path, input_dir='../input'):
    p = Path(input_dir) / path
    cats = json.loads(open(p).read())
    
    return pd.DataFrame([{'category_id': int(cat['id']), 'category': cat['snippet']['title']} for cat in cats['items']])
    
def extract_tag_string(tags):
    return [tag.strip('"') for tag in tags.split('|')]

def load_videos(video_file, category_file, input_dir='../input'):
    p = Path(input_dir) / video_file
    df = pd.read_csv(str(p))
        
    df['tags'] = df['tags'].apply(extract_tag_string)
    
    df['trending_date'] = pd.to_datetime(df.trending_date, format='%y.%d.%m')
    df['publish_time'] = pd.to_datetime(df.publish_time)

    categories = load_category(category_file)
    df = pd.merge(df, categories, on='category_id', how='left')
    return df
df = load_videos('GBvideos.csv', 'GB_category_id.json')


    
print(df.shape)
df.head()
ax = df.groupby('video_id').size().hist(bins=20)
ax.set_xlabel('Days trending')
ax.set_ylabel('Count')
ax = df.description.str.len().plot(kind='hist', bins=30)
ax.set(xlabel='Description length');
df.drop_duplicates('video_id', keep='first').groupby('category').size().sort_values(ascending=False).plot(kind='bar');
videos = df.sort_values('publish_time').drop_duplicates('video_id', keep='last')
RESAMPLE_RATE = '1d'

resampled = (
    videos
        .query('publish_time > "2017-10"')
        .resample(RESAMPLE_RATE, on='publish_time')
        .size()
)

ax = resampled.plot(linestyle=':', linewidth=1, label='Raw')

ROLLING_DAYS = 7
rolling = resampled.rolling(ROLLING_DAYS, center=True)
methods = ['median', 'mean']
for method in methods:
    getattr(rolling, method)().plot(ax=ax, label=f'{RESAMPLE_RATE} rolling {method}')
    
ax.set(xlabel='Publish Time', ylabel='Count')
ax.legend();
g = sns.FacetGrid(videos, row="category", margin_titles=True, aspect=4)
bins = np.linspace(0, 3200000, 50)
axes = g.map(plt.hist, "likes", color="steelblue", bins=bins, lw=0)
# axes.axes[0][0].set_yscale('log', nonposy='clip')
for ax in axes.axes:
    ax[0].set_yscale('log', nonposy='clip');
