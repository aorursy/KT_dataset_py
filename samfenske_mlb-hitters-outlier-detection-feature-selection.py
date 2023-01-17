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
hitters=pd.read_csv('/kaggle/input/hof-data/hitters2.csv').drop(columns='Unnamed: 0')
hitters
names=pd.read_csv('/kaggle/input/the-history-of-baseball/player.csv')
names['name']=names['name_first']+' '+names['name_last']
names=names[['player_id','name']]
names=names[names['player_id'].isin(hitters['player_id'].tolist())]
hitters=hitters.join(names.set_index(['player_id']), on='player_id')
hitters
hitters['avg']=hitters['h']/hitters['ab']
hitters
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(20,10))
sns.heatmap(hitters.corr(),annot=True,linewidth=0.5)
p1=sns.lmplot(x='hr', y="percent", data=hitters,hue='inducted',fit_reg=False,height=4,aspect=4)
ax = p1.axes[0,0]
for i in range(len(hitters)):
    ax.text(hitters['hr'][i], hitters['percent'][i], hitters['name'][i],
           fontsize='small',rotation=45)
p1=sns.lmplot(x='h', y="percent", data=hitters,hue='inducted',fit_reg=False,height=4,aspect=4)
ax = p1.axes[0,0]
for i in range(len(hitters)):
    ax.text(hitters['h'][i], hitters['percent'][i], hitters['name'][i],
           fontsize='small',rotation=45)
outliers=['bondsba01','mcgwima01','sheffga01','palmera01','sosasa01','rosepe01']
hitters_filtered=hitters[-hitters['player_id'].isin(outliers)].reset_index().drop(columns='index')
hitters_filtered
plt.figure(figsize=(20,10))
sns.heatmap(hitters_filtered.corr(),annot=True,linewidth=0.5)
df=pd.DataFrame(hitters_filtered.corr()['percent']).reset_index()
df['Beat Threshold']=abs(df['percent'])>0.45
sns.lmplot(x='index', y="percent", data=df,hue='Beat Threshold',fit_reg=False,height=4,
           aspect=4).set_xticklabels(rotation=90)
def scatter(attribute):
    p1=sns.lmplot(x=attribute, y="percent", data=hitters,hue='inducted',fit_reg=False,height=4,aspect=4)
    ax = p1.axes[0,0]
    for i in range(len(hitters)):
        ax.text(hitters[attribute][i], hitters['percent'][i], hitters['name'][i],
               fontsize='small',rotation=45)
scatter('g')
scatter('ab')
scatter('r')
scatter('h')
scatter('hr')
scatter('rbi')
scatter('bb')
hitters_filtered.to_csv('hitters_filtered')