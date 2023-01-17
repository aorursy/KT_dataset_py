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
music = pd.read_csv("../input/msd-audio-features/year_prediction.csv")
music = music.sort_values(by=['label'])
music['year'] = music['label']
music.head()
music.info()
music['label'].describe()
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option("max_columns", None)
def boxplot(query, y):
    f, ax = plt.subplots(figsize=(20, 9))
    ax = sns.boxplot(x="label", y=y, data=music.query(query))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    plt.show()
    
boxplot('label >= 1925 & label <= 1995', "TimbreAvg7")
music2 = music[["TimbreAvg1", "TimbreAvg2", "TimbreAvg3", "TimbreAvg4","TimbreAvg5", "TimbreAvg6"
              ,"TimbreAvg7","TimbreAvg8","TimbreAvg9","TimbreAvg10","TimbreAvg11","TimbreAvg12"]]
music2.info()
f, ax = plt.subplots(figsize=(20, 9))

ax=sns.boxplot(x="variable", y="value", data=pd.melt(music2))

plt.show()
corr=music2.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
music['year'] = music['label']
f, ax = plt.subplots(figsize=(20, 9))
ax=sns.set_style('darkgrid')
ax=sns.distplot(music['year'])
#group the songs by decade 
music['decade'] = music.label.apply(lambda year : year-(year%10))
music.info()
music4 = music[["decade","TimbreAvg1", "TimbreAvg2", "TimbreAvg3", "TimbreAvg4","TimbreAvg5", "TimbreAvg6"
              ,"TimbreAvg7","TimbreAvg8","TimbreAvg9","TimbreAvg10","TimbreAvg11","TimbreAvg12"]]
columns = music4.groupby(['decade']).mean().columns
labels = ["{:02d}'s".format(l%100) for l in sorted(music4.decade.unique())]
fig, ax = plt.subplots(figsize=(20,5)) 
sns.heatmap(music4.groupby(['decade']).mean().iloc[:,0:20], yticklabels=labels)
plt.ylabel("Release Decade")
plt.xlabel("Features (Mean)")
plt.show()
from sklearn.preprocessing import MinMaxScaler
music3 = pd.DataFrame(MinMaxScaler().fit_transform(music2)).rename(columns={0:'TimbreAvg1',1: 'TimbreAvg2',2: 'TimbreAvg3',
                                                                 3:'TimbreAvg4',4:'TimbreAvg5', 5:'TimbreAvg6',
                                                                 6:'TimbreAvg7',7:'TimbreAvg8',8:'TimbreAvg9',
                                                                 9:'TimbreAvg10', 10:'TimbreAvg11', 11:'TimbreAvg12'})
music3.describe()
import plotly.graph_objects as go


fig = go.Figure(data=
    go.Parcoords(
        #line = dict(color = music['decade'],
                   #colorscale = [[0,'purple'],[0.5,'lightseagreen'],[1,'gold']]),
        dimensions = list([
            dict(range = [0,1],
                constraintrange = [.5,1],
                label = 'TimbreAvg1', values = music3['TimbreAvg1']),
            dict(range = [0,1],
                label = 'TimbreAvg2', values = music3['TimbreAvg2']),
            dict(range = [0,1],
                label = 'TimbreAvg3', values = music3['TimbreAvg3']),
            dict(range = [0,1],
                label = 'TimbreAvg4', values = music3['TimbreAvg4']),
            dict(range = [0,1],
                label = 'TimbreAvg5', values = music3['TimbreAvg5']),
            dict(range = [0,1],
                label = 'TimbreAvg6', values = music3['TimbreAvg6']),
            dict(range = [0,1],
                label = 'TimbreAvg7', values = music3['TimbreAvg7']),
            dict(range = [0,1],
                label = 'TimbreAvg8', values = music3['TimbreAvg8']),
            dict(range = [0,1],
                label = 'TimbreAvg9', values = music3['TimbreAvg9']),
            dict(range = [0,1],
                label = 'TimbreAvg10', values = music3['TimbreAvg10']),
            dict(range = [0,1],
                label = 'TimbreAvg11', values = music3['TimbreAvg11']),
            dict(range = [0,1],
                label = 'TimbreAvg12', values = music3['TimbreAvg12'])
        ])
    )
)

fig.update_layout(
    plot_bgcolor = 'white',
    paper_bgcolor = 'white'
)

fig.show()