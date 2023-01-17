# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import scipy as sc
from scipy import interpolate
import sklearn
import matplotlib.pyplot as plt
import warnings

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm

import networkx as nx
from math import pi

# disable warnings (prevent eruption during some visualisations)
warnings.filterwarnings('ignore')
# set import path
import_path = '../input/'

# Import tables
train_df = pd.read_csv(import_path + 'train.csv', index_col=0)
test_df = pd.read_csv(import_path + 'test.csv', index_col=0)

# display head of table
train_df.head(2)
# Quick insight of the missing values
fig, ax = plt.subplots(2, 1, figsize=(15,5))

titles = ['train', 'test']
for i, table in enumerate([train_df, test_df]):
    x_labels = np.array(list(table.isna().mean().index))[1-i:]
    x_pos = np.arange(len(x_labels))
    y = np.array(list(table.isna().mean()))[1-i:]

    ax[i].set_title('% of missing values per column for {} table.'.format(titles[i].upper()))
    ax[i].set_xticks(x_pos)
    ax[i].set_xticklabels(x_labels)
    ax[i].bar(x_pos, y, width=0.2, edgecolor='black', color=(0.8, 0.3, 0.3))
    ax[i].grid(axis='y')
    ax[i].set_ylim(0, 1)
    ax[i].set_xlabel('Labels')
    ax[i].set_ylabel('% Missing')

    for pos, value in zip(x_pos, y):
        ax[i].annotate(str(round(value*100,2)) + '%',\
                       xy=(pos, value), arrowprops=None,\
                       xytext=(pos - 0.2, value + 0.1),\
                      fontsize=15)
fig.tight_layout()

plt.show()
# 'Sex' feature processing (binary encoding)
train_df['Sex'] = train_df.Sex.apply(lambda name: (name == 'female')*1 - (name == 'male')*1)
# Pie chart, where the slices will be ordered and plotted counter-clockwise:
aggregation = train_df.groupby(['Sex', 'Survived'])['Name'].count().to_dict()

fig, ax = plt.subplots(2, 2, figsize=(8,7))

for i, sex in enumerate([-1, 1]):
    for survived in [0, 1]:
        
        # Define x, y
        x_pos = survived
        y = aggregation[(sex, survived)]
        
        # plot bar
        ax[i, 0].bar(x_pos, y, width=0.1, edgecolor='black', color=(survived, 0.8, 0.9), alpha=0.8)
        
    sex_label = (sex==-1)*'male'.upper() + (sex==1)*'female'.upper()
    ax[i, 0].set_title('Survival Count By Sex Category: ' + sex_label)
    ax[i, 0].legend(['died', 'survived'])
    ax[i, 0].grid(axis='y')
    ax[i, 0].set_ylim(0, 500)
    
# pie chart
colors = [(0, 0.8, 0.9), (1, 0.8, 0.9)]
for i, sex in enumerate([-1, 1]):

    labels = ['died', 'survived']
    sizes = []
    for survived in [0, 1]:
        sizes.append(aggregation[(sex, survived)])       
    # plot pie
    ax[i, 1].pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90, colors=colors)
        
    sex_label = (sex==-1)*'male'.upper() + (sex==1)*'female'.upper()
    ax[i, 1].set_title('Survival Count By Sex Category: ' + sex_label)
    ax[i, 1].legend(['died', 'survived'])
    ax[i, 1].grid(axis='y')
    
fig.tight_layout()
fig.patch.set_facecolor((0.8, 0.8, 0.6))
plt.show()
# Age density by Sex
data = [list(train_df[train_df['Sex']==-1]['Age'].dropna()), list(train_df[train_df['Sex']==1]['Age'].dropna())]
# Build Histogram
pdf_shape = []
for i in range(0, 2):
    y,x = sc.histogram(data[i], bins=10)
    x = x[1:]
    f = sc.interpolate.interp1d(x, y, kind='cubic')

    pdf_shape.append(f)
    
fig, ax = plt.subplots(2, 2, figsize=(10,7))
colors = [(1, 0.8, 0.9), (0, 0.8, 0.9)]
labels = ['Male', 'Female']

for i in range(0, 2):
    ax[0, i].hist(data[i], bins=10, alpha=0.8, color=colors[0])
    f = pdf_shape[i]
    x = np.linspace(f.x.min(), f.x.max(), 100)
    ax[0, i].plot(x, f(x), color=colors[1], linewidth=3)
    ax[0, i].set_xlim(0, 80)
    ax[0, i].grid(axis='y')
    ax[0, i].set_title('Density of Age for Sex Category ' + labels[i])

for i in range(0, 2):
    ax[1, i].boxplot(data[i])
    ax[1, i].grid(axis='y')
    ax[1, i].set_title('BoxPlot of Age for Sex Category ' + labels[i])
    ax[1, i].set_ylim(0, 90)

fig.tight_layout()
fig.patch.set_facecolor((0.7, 1, 1))

plt.show()
sub_df = train_df[['Age', 'Fare', 'Survived']].dropna()
sub_df['Age'] = sub_df['Age'].map(lambda x: 5*(x//5))
sub_df['Fare'] = sub_df['Fare'].map(lambda x: 10*(x//10))

dictionnaire = sub_df.groupby(['Fare', 'Age'])['Survived'].count().to_dict()

def get_count(i, j, dictionnaire):
    if (i,j) in list(dictionnaire.keys()):
        # (i,j) is a key
        return dictionnaire[(i,j)]
    else:
        return 0

from mpl_toolkits.mplot3d import Axes3D

# setup the figure and axes
fig = plt.figure(figsize=(15, 10))

ax = fig.add_subplot(111, projection='3d')
_x = np.array([i[0] for i in list(dictionnaire.keys())])
_y = np.array([i[1] for i in list(dictionnaire.keys())])

np.random.shuffle(_x)
np.random.shuffle(_y)

_x = _x[:40]
_y = _y[:40]

_xx, _yy = np.meshgrid(_x, _y)
x, y = _xx.ravel(), _yy.ravel()

top = np.array([get_count(i, j, dictionnaire) for i, j in zip(x, y)])
bottom = np.zeros_like(top)
width = 4
depth = 2

ax.bar3d(x, y, bottom, width, depth, top, alpha=0.3, color='red')

ax.set_title('3D Plot #Passengers Count = f(Fare, Age)')
ax.set_xlabel('Fare')
ax.set_ylabel('Age')
ax.set_zlabel('#Passengers Count')

plt.show()
dictionnaire = train_df.groupby(['Embarked', 'Pclass', 'Sex'])['Name'].count().to_dict()

embarked_scope = ['C','S','Q']
class_scope = [1, 2, 3]
survived_scope = [-1, 1]

colors = [(0, 0, 0), (1, 0, 0)]

fig, axes = plt.subplots(3, 3, figsize=(9,9))
for i, embarked_ in enumerate(embarked_scope):
    for j, class_ in enumerate(class_scope):
        
        a = dictionnaire[(embarked_, class_, -1)]
        b = dictionnaire[(embarked_, class_, 1)]
        
        axes[i, j].bar(0, a, width=0.1, edgecolor='black', color=colors[0], alpha=0.8)
        axes[i, j].bar(1, b, width=0.1, edgecolor='black', color=colors[1], alpha=0.8)
        
        axes[i, j].text(0.5*0.75, 1.2*max(a, b)/2 , '#Passengers:', fontsize=10, color='black',  alpha=1)
        axes[i, j].text(0.5, max(a, b)/2, str(a+b), fontsize=20, color='black',  alpha=1)
        
        axes[i, j].text(0.5*0.75, 0.8*max(a, b)/2 , 'Death Rate:', fontsize=10, color='black',  alpha=1)
        axes[i, j].text(0.5, 0.6*max(a, b)/2, str(round(100*a/(a+b)))+'%', fontsize=20, color='black',  alpha=1)
        
        if j==0:
            axes[i, j].legend(['Died', 'Survived'], loc='upper center')

        axes[i, j].set_title(str(embarked_) + ':' + str(class_), fontsize=20)
        axes[i, j].grid(axis='y')
        
fig.tight_layout()
plt.show()
fig, ax = plt.subplots(1, 3, figsize=(15,7))

for i, label_embarked in enumerate(np.unique(list(train_df['Embarked'].dropna()))):
    sub_table = train_df[train_df['Embarked']==label_embarked][train_df['Survived']==0]
    x = sub_table['Age']
    y = sub_table['Fare']
    scale = np.ones(len(x))*150
    ax[i].axvline(15, color='black', alpha=0.7, lw=3, label='Youth')
    ax[i].scatter(x, y, c='red', s=scale, alpha=0.8, edgecolors='black', label='died')

colors = ['blue', 'pink']
for i, label_embarked in enumerate(np.unique(list(train_df['Embarked'].dropna()))):
    sub_table = train_df[train_df['Embarked']==label_embarked]
    for j, sex_cat in enumerate([-1, 1]):
        x = sub_table[sub_table['Sex']==sex_cat]['Age']
        y = sub_table[sub_table['Sex']==sex_cat]['Fare']
        scale = np.ones(len(x))*50
        cat_label = 'men'*(sex_cat==-1) + 'women'*(sex_cat==1)
        ax[i].scatter(x, y, c=colors[j], s=scale, alpha=0.8, edgecolors='black', label=label_embarked + ' ' + cat_label)
        ax[i].set_xlabel('Age')
        ax[i].set_ylabel('Fare')
        ax[i].grid(True)
    ax[i].legend()
    ax[i].set_ylim(0, 20)


plt.show()
embarked_scope = ['C','S','Q']
class_scope = [1, 2, 3]

colors = [(0, 0, 0), (1, 0, 0)]

fig, axes = plt.subplots(3, 3, figsize=(12,12))
for i, embarked_ in enumerate(embarked_scope):
    for j, class_ in enumerate(class_scope):
        
        sub_df = train_df[train_df['Embarked']==embarked_][train_df['Pclass']==class_][['Age', 'Fare', 'Sex', 'Survived']].dropna()
        
        x_male_died = sub_df[sub_df['Survived']==0][sub_df['Sex']==-1]['Age']
        y_male_died = sub_df[sub_df['Survived']==0][sub_df['Sex']==-1]['Fare']
        
        x_female_died = sub_df[sub_df['Survived']==0][sub_df['Sex']==1]['Age']
        y_female_died = sub_df[sub_df['Survived']==0][sub_df['Sex']==1]['Fare']
        
        x_male = sub_df[sub_df['Sex']==-1]['Age']
        y_male = sub_df[sub_df['Sex']==-1]['Fare']
        
        x_female = sub_df[sub_df['Sex']==1]['Age']
        y_female = sub_df[sub_df['Sex']==1]['Fare']
        
        scale = np.ones(len(x_male_died))*150
        axes[i, j].scatter(x_male_died, y_male_died, c='red', s=scale, alpha=0.8, edgecolors='black', label='died')
        axes[i, j].scatter(x_female_died, -y_female_died, c='red', s=scale, alpha=0.8, edgecolors='black')
        
        
        scale = np.ones(len(x_male))*50
        axes[i, j].scatter(x_male, y_male, c='blue', s=scale, alpha=0.8, edgecolors='black', label='male')
        
        scale = np.ones(len(x_female))*50
        axes[i, j].scatter(x_female, -y_female, c='pink', s=scale, alpha=0.8, edgecolors='black', label='female')
        axes[i, j].axvline(16, lw=3, alpha=0.6, color='black')
        axes[i, j].axvline(65, lw=3, alpha=0.6, color='black')
        axes[i, j].axhline(0, lw=3, alpha=1, color='pink')
        axes[i, j].set_title('Embarked:{}, Class:{}'.format(embarked_, class_))
        
        if i == 2:
            axes[i, j].legend(loc='lower right')
        axes[i, j].grid()
        axes[i, j].set_xlabel('Age')
        axes[i, j].set_ylabel('Fare')

fig.tight_layout()
plt.show()
colors = [(0, 0, 0), (1, 0, 0)]

fig = plt.figure(figsize=(10,10))

ax1 = plt.subplot2grid((4, 4), (1, 1), colspan=2, rowspan=2)

embarked_ = 'S'
class_ = 3

sub_df = train_df[train_df['Embarked']==embarked_][train_df['Pclass']==class_][['Age', 'Fare', 'Sex', 'Survived']].dropna()
        
x_male_died = sub_df[sub_df['Survived']==0][sub_df['Sex']==-1]['Age']
y_male_died = sub_df[sub_df['Survived']==0][sub_df['Sex']==-1]['Fare']

x_female_died = sub_df[sub_df['Survived']==0][sub_df['Sex']==1]['Age']
y_female_died = sub_df[sub_df['Survived']==0][sub_df['Sex']==1]['Fare']

x_male = sub_df[sub_df['Sex']==-1]['Age']
y_male = sub_df[sub_df['Sex']==-1]['Fare']

x_female = sub_df[sub_df['Sex']==1]['Age']
y_female = sub_df[sub_df['Sex']==1]['Fare']
        
scale = np.ones(len(x_male_died))*150
ax1.scatter(x_male_died, y_male_died, c='red', s=scale, alpha=0.8, edgecolors='black', label='died')
ax1.scatter(x_female_died, -y_female_died, c='red', s=scale, alpha=0.8, edgecolors='black')
                
scale = np.ones(len(x_male))*50
ax1.scatter(x_male, y_male, c='blue', s=scale, alpha=0.8, edgecolors='black', label='male')

scale = np.ones(len(x_female))*50
ax1.scatter(x_female, -y_female, c='pink', s=scale, alpha=0.8, edgecolors='black', label='female')
ax1.axvline(16, lw=3, alpha=0.6, color='black')
ax1.axvline(65, lw=3, alpha=0.6, color='black')
ax1.axhline(0, lw=3, alpha=1, color='pink')
ax1.set_title('Embarked:{}, Class:{}'.format(embarked_, class_))

ax1.text(60, 20, 'male', style='italic',
        bbox={'facecolor':'white', 'alpha':0.5, 'pad':3})
ax1.text(60, -20, 'female', style='italic',
        bbox={'facecolor':'white', 'alpha':0.5, 'pad':3})

ax1.legend()
ax1.grid()
ax1.set_xlabel('Age')
ax1.set_ylabel('Fare')

ax2 = plt.subplot2grid((4, 4), (0, 1), colspan=2)
ax2.hist(x_male, bins=20, range=[0, 80], color='white', edgecolor='black')
ax2.hist(x_male_died, bins=20, range=[0, 80], color=colors[1], alpha=0.6)

ax3 = plt.subplot2grid((4, 4), (3, 1), colspan=2)
ax3.hist(x_female, bins=20, range=[0, 80], color='white', edgecolor='black', label='count')
ax3.hist(x_female_died, bins=20, range=[0, 80], color=colors[1], alpha=0.6, label='died')
ax3.legend()
ax3.invert_yaxis()

ax4 = plt.subplot2grid((4, 4), (1, 0))
ax4.hist(y_male, bins=20, orientation="horizontal", color='white', edgecolor='black')
ax4.hist(y_male_died, bins=20, orientation="horizontal", color=colors[1], alpha=0.6)
ax4.invert_xaxis()

ax5 = plt.subplot2grid((4, 4), (2, 0))
ax5.hist(-y_female, bins=20, orientation="horizontal", color='white', edgecolor='black')
ax5.hist(-y_female_died, bins=20, orientation="horizontal", color=colors[1], alpha=0.6)
ax5.invert_xaxis()

ax6 = plt.subplot2grid((4, 4), (1, 3), rowspan=2)
ax6.hist(np.concatenate((y_male, -y_female), axis=0), bins=20, orientation="horizontal", color='white', edgecolor='black')
ax6.hist(np.concatenate((y_male_died, -y_female_died), axis=0), bins=20, orientation="horizontal", color=colors[1], alpha=0.6)

fig.tight_layout()
plt.show()
import re

def parse_name(sample_name):
    dictionnary = {}
    dictionnary['LastName'] = sample_name.split(' ')[0][:-1]
    dictionnary['Title'] = sample_name.split(' ')[1]
    if dictionnary['Title'] == 'Mrs.':
        if '(' in sample_name:
            original_name = re.search('(\(.*?)\)', sample_name).group()[1:-1]
    return dictionnary


train_df['LastName'] = train_df['Name'].map(lambda name: parse_name(name)['LastName'])
train_df['Title'] = train_df['Name'].map(lambda name: parse_name(name)['Title'])
dict_edges = {}

# Nodes
passenger_ids = list(train_df.index)
start_index = passenger_ids[0]
end_index = passenger_ids[-1]

def get_table_lastName(table_df, name):
    return table_df[table_df['LastName']==name]

def vec_distance_measure(vec1, vec2):
    vecs = []
    for vec in [vec1, vec2]:
        vec = list(vec)
        vec = np.array([vec[1], vec[5] + vec[6], vec[7], vec[8], vec[10], vec[11]])
        vecs.append(vec)
    return (1 - np.mean(vecs[0] == vecs[1]))*8 + 5

for lastname in np.unique(train_df['LastName']):

    sub_table = get_table_lastName(train_df, lastname)
    
    indexes = list(sub_table.index)
    for i, idx_i in enumerate(indexes):
        for j in range(i+1, len(indexes)):
            idx_j = indexes[j]
            dict_edges[(idx_i, idx_j)] = vec_distance_measure(sub_table.loc[idx_i], sub_table.loc[idx_j])
e = []
for key_ in list(dict_edges.keys()):
    weight = dict_edges[key_]
    e.append((key_[0], key_[1], {'weight': weight/30}))

G = nx.Graph(e)

node_colors = []
for node in G:
    if train_df.iloc[node]['Survived'] == 1:
        color = 'blue'
    else:
        color = 'red'
    node_colors.append(color)

node_sizes = []
for node in G:
    size_ = (train_df.iloc[node]['Age'] < 20)*20 +  (train_df.iloc[node]['Age'] >= 20)*100 + 100
    node_sizes.append(size_)


plt.figure(figsize=(15,15))
pos = nx.spring_layout(G)
nx.draw(G, pos= pos, node_color=node_colors, node_size=node_sizes,\
        with_labels = False, edge_color='black', alpha=0.7)

print('Titanic Graph Visualisation of families')
print('Survived: Blue\nDied: Red')
plt.show()
