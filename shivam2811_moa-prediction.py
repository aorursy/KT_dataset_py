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
import pandas as pd
import numpy as np
from IPython.display import display
import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import time
# for visualizing

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import plotly.express as px

# for adding extra statistical stuff

from scipy.stats import skew, norm
train_feat = pd.read_csv('../input/lish-moa/train_features.csv')
train_target = pd.read_csv('../input/lish-moa/train_targets_scored.csv')

test_feat = pd.read_csv('../input/lish-moa/test_features.csv')

print('Train Feature Samples:')
train_feat.head(3)
print('Test Feature Samples:')
test_feat.head(3)
print('Number of rows in training set: ', train_feat.shape[0])
print('Number of columns in training set: ', train_feat.shape[1])
print('Number of rows in test set: ', test_feat.shape[0])
print('Number of columns in test set: ', test_feat.shape[1])
train_miss=train_feat.isnull().sum().sum()
test_miss=train_feat.isnull().sum().sum()
print('Number of Null values in training set: ',train_miss)
print('Number of Null values in training set: ',train_miss)
train_feat.info()
plt.style.use('ggplot')

fig = plt.figure(constrained_layout=True, figsize=(20, 12))


grid = gridspec.GridSpec(ncols=6, nrows=3, figure=fig)

ax1 = fig.add_subplot(grid[0, :3])

ax1.set_title(f'Train cp_type Distribution',weight='bold')

sns.countplot(x='cp_type',
                    data=train_feat,
                    palette="rocket",
                    ax=ax1,
                    order=train_feat['cp_type'].value_counts().index)

total = float(len(train_feat['cp_type']))


for p in ax1.patches:
    height = p.get_height()
    ax1.text(p.get_x() + p.get_width() / 2.,
            height + 2,
            '{:1.2f}%'.format((height / total) * 100),
            ha='center')


ax2 = fig.add_subplot(grid[0, 3:])



sns.countplot(x='cp_type',
                    data=test_feat,
                    palette="rocket",
                    ax=ax2,
                    order=test_feat['cp_type'].value_counts().index)

total = float(len(test_feat['cp_type']))

ax2.set_title(f'Test cp_type Distribution', weight='bold')


for p in ax2.patches:
    height = p.get_height()
    ax2.text(p.get_x() + p.get_width() / 2.,
            height + 2,
            '{:1.2f}%'.format((height / total) * 100),
            ha='center')
ax3 = fig.add_subplot(grid[1, :3])

ax3.set_title(f'Train cp_time Distribution', weight='bold')

sns.countplot(x='cp_time',
                    data=train_feat,
                    palette="rocket",
                    ax=ax3,
                    order=train_feat['cp_time'].value_counts().index)

total = float(len(train_feat['cp_time']))


for p in ax3.patches:
    height = p.get_height()
    ax3.text(p.get_x() + p.get_width() / 2.,
            height + 2,
            '{:1.2f}%'.format((height / total) * 100),
            ha='center')

ax4 = fig.add_subplot(grid[1, 3:])

ax4.set_title(f'Test cp_time Distribution', weight='bold')

sns.countplot(x='cp_time',
                    data=test_feat,
                    palette="rocket",
                    ax=ax4,
                    order=train_feat['cp_time'].value_counts().index)

total = float(len(test_feat['cp_time']))


for p in ax4.patches:
    height = p.get_height()
    ax4.text(p.get_x() + p.get_width() / 2.,
            height + 2,
            '{:1.2f}%'.format((height / total) * 100),
            ha='center')
    
ax5 = fig.add_subplot(grid[2, :3])

ax5.set_title(f'Train cp_dose Distribution', weight='bold')

sns.countplot(x='cp_dose',
                    data=train_feat,
                    palette="rocket",
                    ax=ax5,
                    order=train_feat['cp_dose'].value_counts().index)

total = float(len(train_feat['cp_dose']))


for p in ax5.patches:
    height = p.get_height()
    ax5.text(p.get_x() + p.get_width() / 2.,
            height + 2,
            '{:1.2f}%'.format((height / total) * 100),
            ha='center')

ax6 = fig.add_subplot(grid[2, 3:])

ax6.set_title(f'Test cp_dose Distribution', weight='bold')

sns.countplot(x='cp_dose',
                    data=test_feat,
                    palette="rocket",
                    ax=ax6,
                    order=train_feat['cp_dose'].value_counts().index)

total = float(len(test_feat['cp_dose']))


for p in ax6.patches:
    height = p.get_height()
    ax6.text(p.get_x() + p.get_width() / 2.,
            height + 2,
            '{:1.2f}%'.format((height / total) * 100),
            ha='center')
def plot_distplot(f1):
    plt.style.use('seaborn')
    sns.set_style('whitegrid')

    fig= plt.figure(figsize=(20,20))
    #2 rows 2 cols
    #first row, first col
    for i in range(0,12):
        ax1 = plt.subplot2grid((5,4),((i // 3) + 1, (i % 3) + 1))
        sns.distplot(train_feat[f1])
        plt.title(f1[i],weight='bold', fontsize=12)
        plt.yticks(weight='bold')
        plt.xticks(weight='bold')
    return plt.show()
train_columns = train_feat.columns.to_list()
g_list = [i for i in train_columns if i.startswith('g-')]
c_list = [i for i in train_columns if i.startswith('c-')]
c_plot= [c_list[random.randint(0, len(c_list)-1)] for i in range(50)]
plot_distplot(c_plot[:12])
cells=train_feat[c_list]
#Plot heatmap
plt.figure(figsize=(25,12))
sns.heatmap(cells.corr(), cmap='coolwarm', alpha=0.9)
plt.title('Correlation: Cell', fontsize=15, weight='bold')
plt.xticks(weight='bold')
plt.yticks(weight='bold')
plt.show()
correlations = cells.corr().abs().unstack().sort_values(kind="quicksort",ascending=False).reset_index()
correlations = correlations[correlations['level_0'] != correlations['level_1']] #preventing 1.0 corr
corr_max=correlations.level_0.head(150).tolist()
corr_max=list(set(corr_max)) #removing duplicates

corr_min=correlations.level_0.tail(50).tolist()
corr_min=list(set(corr_min)) #removing duplicates
correlations.head()
correlations.tail()
g_plot= [g_list[random.randint(0, len(g_list)-1)] for i in range(50)]
plot_distplot(g_plot[:12])
geans=train_feat[g_list]
#Plot heatmap
plt.figure(figsize=(25,12))
sns.heatmap(geans.corr(), cmap='coolwarm', alpha=0.9)
plt.title('Correlation: Cell', fontsize=15, weight='bold')
plt.xticks(weight='bold')
plt.yticks(weight='bold')
plt.show()
correlations = geans.corr().abs().unstack().sort_values(kind="quicksort",ascending=False).reset_index()
correlations = correlations[correlations['level_0'] != correlations['level_1']] #preventing 1.0 corr
corr_max=correlations.level_0.head(150).tolist()
corr_max=list(set(corr_max)) #removing duplicates

corr_min=correlations.level_0.tail(50).tolist()
corr_min=list(set(corr_min)) #removing duplicates
correlations.head()
correlations.tail()
print('Train Target Samples:')
display(train_target.head(3))
train_target.describe()
x = train_target.drop(['sig_id'], axis=1).sum(axis=0).sort_values().reset_index()
x.columns = ['column', 'nonzero_records']

fig = px.bar(
    x.tail(50), 
    x='nonzero_records', 
    y='column',
    title='Columns with the higher number of positive samples (top 50)', 
    height=1000, 
    width=800,
    color='nonzero_records'
)

fig.show()
fig = px.bar(
    x.head(50), 
    x='nonzero_records', 
    y='column', 
    title='Columns with the lowest number of positive samples (top 50)', 
    height=1000, 
    width=800,
    color='nonzero_records'
)

fig.show()