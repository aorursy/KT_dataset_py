import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

sns.set_palette('Set3') # You can use plt.style.use() function

import os

os.listdir('../input/students-performance-in-exams')
data = pd.read_csv('../input/students-performance-in-exams/StudentsPerformance.csv')
data.head()
data.info()
data.describe(include='all')
data.corr()
data.isnull().sum()
ax = sns.countplot(data['gender'])

ax.set_title('Gender Ratio')
fig, ax = plt.subplots(1, 1, figsize=(7, 7))

sns.barplot(data['gender'].value_counts().index, data['gender'].value_counts(), ax=ax)
def barp_xyz(x, y, hue):

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    sns.barplot(x, y, hue=hue, data=data)
for i in ['writing score', 'reading score', 'math score']:

    barp_xyz('parental level of education' ,i,'gender')
def barp_xyz_with_avgline(x, y, hue, ax):

    sns.barplot(x, y, hue=hue, data=data, ax=ax)

    avg = (data[y].sum().sum())/len(data)

    ax.axhline(avg, ls='--', color='r')

    ax.set_xticklabels(data[x].value_counts().index, rotation=30)

    ax.set_title(f'The Average Score of {y} is {avg}')
fig, ax = plt.subplots(1, 3, figsize=(27, 6))

for idx, i in enumerate(['writing score', 'reading score', 'math score']):

    barp_xyz_with_avgline('parental level of education' ,i,'gender', ax[idx])
def distp_xyz(data, x,ax):

    sns.distplot(data[x], ax=ax)
fig, ax = plt.subplots(1, 3, figsize=(27, 6))

for idx, i in enumerate(['writing score', 'reading score', 'math score']):

    distp_xyz(data, i, ax[idx])
def kdep_xyz(data, x,ax):

    sns.distplot(data[x], ax=ax)



fig, ax = plt.subplots(1, 3, figsize=(27, 6))

for idx, i in enumerate(['writing score', 'reading score', 'math score']):

    kdep_xyz(data, i, ax[idx])
def catp_xyz_with_avgline(x, y, hue, ax, kind):

    sns.catplot(x, y, hue=hue, data=data, ax=ax, kind=kind)

    #avg = (data[y].sum().sum())/len(data)

    #ax.axhline(avg, ls='--', color='r')

    ax.set_xticklabels(data[x].value_counts().index, rotation=30)

    ax.set_title(f'{y} : {kind} plot.')

    plt.close(2) # trick to hide empty plot
fig, ax = plt.subplots(1, 3, figsize=(27, 6))

for idx, i in enumerate(['writing score', 'reading score', 'math score']):

    catp_xyz_with_avgline('parental level of education' ,i,'gender', ax[idx], kind='swarm')
fig, ax = plt.subplots(1, 3, figsize=(27, 6))

for idx, i in enumerate(['writing score', 'reading score', 'math score']):

    catp_xyz_with_avgline('parental level of education' ,i,'gender', ax[idx], kind='violin')
    

fig, ax = plt.subplots(1, 3, figsize=(27, 6))

for idx, i in enumerate(['writing score', 'reading score', 'math score']):

    catp_xyz_with_avgline('parental level of education' ,i,'gender', ax[idx], kind='box')
    

fig, ax = plt.subplots(1, 3, figsize=(27, 6))

for idx, i in enumerate(['writing score', 'reading score', 'math score']):

    catp_xyz_with_avgline('parental level of education' ,i,'gender', ax[idx], kind='boxen')
data['race/ethnicity'].unique()
fig, ax = plt.subplots(figsize=(27, 9))

data_groupb_math = data[data['race/ethnicity']=='group B']['math score']

data_groupb_read = data[data['race/ethnicity']=='group B']['reading score']

sns.pointplot(x=np.arange(len(data_groupb_math)), y=data_groupb_math, color=sns.color_palette("Set2")[0], ax=ax)

sns.pointplot(x=np.arange(len(data_groupb_read)), y=data_groupb_read, color= sns.color_palette("Set2")[1], ax=ax)

plt.setp(ax.collections, alpha=0.6) # point alpha

plt.setp(ax.lines, alpha=0.6) # line alpha

plt.show() # This command can ignore plt text output
fig, ax = plt.subplots(1, 2, figsize=(16, 8))

sns.pointplot(x='reading score', y='writing score', hue='gender', data=data, ax=ax[0])

sns.pointplot(x='reading score', y='math score', hue='gender', data=data, ax=ax[1])

plt.setp(ax[0].collections, alpha=0.6) # point alpha

plt.setp(ax[0].lines, alpha=0.6) # line alpha

plt.setp(ax[1].collections, alpha=0.6) # point alpha

plt.setp(ax[1].lines, alpha=0.6) # line alpha

plt.show()
ax = sns.lmplot(x='reading score', y='writing score', hue='gender', data=data, palette='Set1')



plt.show()
ax = sns.lmplot(x='reading score', y='writing score', hue='gender', col='gender', data=data, palette='Set1')



plt.show()
ax = sns.lmplot(x='reading score', y='writing score', hue='gender', col='parental level of education', col_wrap=2, data=data, palette='Set1')



plt.show()
sns.set({'figure.figsize':(7,7)})

g = sns.jointplot('math score', 'reading score', data=data, kind='kde', space=0)

plt.tight_layout()

plt.show()
sns.set({'figure.figsize':(7,7)})

g = sns.jointplot('math score', 'reading score', data=data, kind='scatter', space=0, alpha=0.25, 

                  color=sns.color_palette("Set2")[0]).plot_joint(sns.kdeplot, zorder=0, n_level=6)

plt.tight_layout()

plt.show()
sns.heatmap(data.corr(),linewidths=.5, annot=True, fmt="f", cmap="YlGnBu")

plt.show()
sns.clustermap(data.corr(),linewidths=.5, annot=True, fmt="f", cmap="mako")

plt.show()
sns.pairplot(data)

plt.show()
sns.pairplot(data, hue='gender')

plt.show()
g = sns.PairGrid(data)

g.map(plt.scatter)

plt.show()
g = sns.PairGrid(data, hue='gender')

g.map_upper(plt.scatter)

g.map_diag(sns.kdeplot)

g.map_lower(sns.kdeplot, shadow=True)

plt.show()