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
pd.set_option('display.max_columns', None)

pitchers=pd.read_csv('/kaggle/input/pitching/pitching2.csv').drop(columns='Unnamed: 0')

pitchers
names=pd.read_csv('/kaggle/input/the-history-of-baseball/player.csv')

names['name']=names['name_first']+' '+names['name_last']

names=names[['player_id','name']]

names=names.rename(columns={'player_id':'playerID'})

names=names[names['playerID'].isin(pitchers['playerID'].tolist())]

pitchers=pitchers.join(names.set_index(['playerID']), on='playerID')

pitchers
import matplotlib.pyplot as plt

import seaborn as sns

plt.figure(figsize=(20,10))

sns.heatmap(pitchers.corr(),annot=True,linewidth=0.5)
df=pd.DataFrame(pitchers.corr()['percent']).reset_index()

df['Beat Threshold']=abs(df['percent'])>0.45

df.plot(x='index',y='percent',kind='scatter',rot=90,grid=True)
sns.lmplot(x='index', y="percent", data=df,hue='Beat Threshold',fit_reg=False,height=4,

           aspect=4).set_xticklabels(rotation=90)
def scatter(attribute,show_annotations):

    if show_annotations==False:

        sns.lmplot(x=attribute, y="percent", data=pitchers,hue='inducted',fit_reg=False,size=8,aspect=2)

    else:

        p1=sns.lmplot(x=attribute, y="percent", data=pitchers,hue='inducted',fit_reg=False,size=8,aspect=2)

        ax = p1.axes[0,0]

        for i in range(len(pitchers)):

            ax.text(pitchers[attribute][i], pitchers['percent'][i],pitchers['name'][i],

                   fontsize='small',rotation=45)

        plt.show()
scatter('SV',True)
scatter('BAOpp',True)
pitchers[pitchers['BAOpp']>.4]
for row in range(len(pitchers)):

    if pitchers['BAOpp'][row]>0.4:

        pitchers.iloc[row,14]=None

pitchers
pitchers.corr()['BAOpp']
scatter('BAOpp',True)
scatter('W',True)
pitchers_filtered=pitchers[pitchers['playerID'].isin(['clemero02'])].reset_index().drop(columns='index')
pitchers_filtered.to_csv('pitchers_filtered')