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
pwd
import os
os.chdir(r'/kaggle/input/ted-talks')
import pandas as pd
df = pd.read_csv('ted_main.csv')
df.info()
df2 = pd.read_csv('transcripts.csv')
df2.info()
df.head()
df2.head()
df['views'].max()
df[df['views'] == df['views'].max()]
type(df['film_date'].iloc[0])
import datetime
datetime.datetime.fromtimestamp(1140825600)
datetime.date.fromtimestamp(1140825600)
df['film_date'] = df['film_date'].apply(lambda x: datetime.date.fromtimestamp(x))
df['film_date'].head()
tenpop = df[['main_speaker','title','film_date','views']].sort_values('views',ascending=False).head(10)  #10 most popular talks of all time
tenpop
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
tenpop['abbr'] = tenpop['main_speaker'].apply(lambda x: x[:3])
tenpop
sns.barplot(x='abbr',y='views',data=tenpop)
#plt.figure(figsize=(12,6))
plt.title('Top 10 most viewed TED talks')
plt.tight_layout()
sns.scatterplot(x='views',y='comments',data=df)
plt.title('Scatter relation between views and comments')
df[['views','comments']].corr()
df[['main_speaker','views','title','comments']].sort_values('comments',ascending=False).head(10)
df['disquo'] = df['comments']/df['views']
df[['main_speaker','title','views','comments','disquo']].sort_values('disquo',ascending=False).head(10)
df['month'] = df['film_date'].apply(lambda x: x.month)
df['month'].head(3)
sns.barplot(x='month',y='views',data=df)
plt.title('BarPlot between month and views')
df['year'] = df['film_date'].apply(lambda x: x.year)
df['year'].head(3)
sns.lineplot(x='year',y='views',data=df)
plt.title('Barplot between year and views')
plt.tight_layout()
