import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import missingno as msno 



import chart_studio.plotly as py

import cufflinks as cf

import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots



import os 



print(os.listdir('../input/cat-in-the-dat-ii'))
# matplotlib setting

plt.rc('font', size=12) 

plt.rc('axes', titlesize=14)

plt.rc('axes', labelsize=12) 

plt.rc('xtick', labelsize=12)

plt.rc('ytick', labelsize=12) 

plt.rc('legend', fontsize=12) 

plt.rc('figure', titlesize=14) 

plt.rcParams['figure.dpi'] = 300

sns.set_style("whitegrid")



colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple"]

sns.set_palette(sns.xkcd_palette(colors))
train = pd.read_csv('../input/cat-in-the-dat-ii/train.csv')

test = pd.read_csv('../input/cat-in-the-dat-ii/test.csv')

train.head()
target, train_id = train['target'], train['id']

test_id = test['id']

train.drop(['id'], axis=1, inplace=True)

test.drop(['id'], axis=1, inplace=True)

print(train.shape)

print(test.shape)
print(train.columns)
msno.matrix(train)
msno.matrix(train, sort='ascending')
null_rate = [train[i].isna().sum() / len(train) for i in train.columns]

fig, ax = plt.subplots(1,1,figsize=(20, 7))

sns.barplot(x=train.columns, y=null_rate, ax=ax,color='gray')

ax.set_title("Missing Value Rate (Train)")

ax.set_xticklabels(train.columns, rotation=40)

ax.axhline(y=0.03, color='red')

plt.show()
null_rate = [test[i].isna().sum() / len(train) for i in test.columns]

fig, ax = plt.subplots(1,1,figsize=(20, 7))

sns.barplot(x=test.columns, y=null_rate, ax=ax,color='gray')

ax.set_title("Missing Value Rate (Test)")

ax.set_xticklabels(test.columns, rotation=40)

ax.axhline(y=0.02, color='red')

plt.show()
target_dist = target.value_counts()



fig, ax = plt.subplots(1, 1, figsize=(8,5))



barplot = plt.bar(target_dist.index, target_dist, color = 'lightgreen', alpha = 0.8)

barplot[1].set_color('darkred')



ax.set_title('Target Distribution')

ax.annotate("percentage of target 1 : {}%".format(target.sum() / len(target)),

              xy=(0, 0),xycoords='axes fraction', 

              xytext=(0,-50), textcoords='offset points',

              va="top", ha="left", color='grey',

              bbox=dict(boxstyle='round', fc="w", ec='w'))



plt.xlabel('Target', fontsize = 12, weight = 'bold')

plt.show()
fig, ax = plt.subplots(1,5, figsize=(30, 8))

for i in range(5): 

    sns.countplot(f'bin_{i}', data= train, ax=ax[i])

    ax[i].set_ylim([0, 600000])

    ax[i].set_title(f'bin_{i}', fontsize=15)

fig.suptitle("Binary Feature Distribution (Train Data)", fontsize=20)

plt.show()
fig, ax = plt.subplots(1,5, figsize=(30, 8))

for i in range(5): 

    sns.countplot(f'bin_{i}', data= test, ax=ax[i], alpha=0.7,

                 order=test[f'bin_{i}'].value_counts().index)

    ax[i].set_ylim([0, 600000])

    ax[i].set_title(f'bin_{i}', fontsize=15)

fig.suptitle("Binary Feature Distribution (Test Data)", fontsize=20)

plt.show()
fig, ax = plt.subplots(1,5, figsize=(30, 8))

for i in range(5): 

    sns.countplot(f'bin_{i}', hue='target', data= train, ax=ax[i])

    ax[i].set_ylim([0, 500000])

    ax[i].set_title(f'bin_{i}', fontsize=15)

fig.suptitle("Binary Feature Distribution (Train Data)", fontsize=20)

plt.show()
fig, ax = plt.subplots(2,3, figsize=(30, 15))

for i in range(5): 

    sns.countplot(f'nom_{i}', data= train, ax=ax[i//3][i%3],

                 order=train[f'nom_{i}'].value_counts().index)

    ax[i//3][i%3].set_ylim([0, 350000])

    ax[i//3][i%3].set_title(f'nom_{i}', fontsize=15)

fig.suptitle("Nominal Feature Distribution (Train Data)", fontsize=20)

plt.show()
fig, ax = plt.subplots(2,3, figsize=(30, 15))

for i in range(5): 

    sns.countplot(f'nom_{i}', data= test, ax=ax[i//3][i%3],

                 order=test[f'nom_{i}'].value_counts().index,

                 alpha=0.7)

    ax[i//3][i%3].set_ylim([0, 250000])

    ax[i//3][i%3].set_title(f'nom_{i}', fontsize=15)

fig.suptitle("Nominal Feature Distribution (Test Data)", fontsize=20)

plt.show()
fig, ax = plt.subplots(2,3, figsize=(30, 15))

for i in range(5): 

    sns.countplot(f'nom_{i}', hue='target', data= train, ax=ax[i//3][i%3],

                 order=train[f'nom_{i}'].value_counts().index)

    ax[i//3][i%3].set_ylim([0, 300000])

    ax[i//3][i%3].set_title(f'nom_{i}', fontsize=15)

fig.suptitle("Nominal Feature Distribution (Train Data)", fontsize=20)

plt.show()
for i in range(5):

    data = train[[f'nom_{i}', 'target']].groupby(f'nom_{i}')['target'].value_counts().unstack()

    data['rate'] = data[1]  / (data[0] + data[1] )

    data.sort_values(by=['rate'], inplace=True)

    display(data.style.highlight_max(color='lightgreen').highlight_min(color='#cd4f39'))
train[[f'nom_{i}' for i in range(5, 10)]].describe(include='O')
fig, ax = plt.subplots(2,1, figsize=(30, 10))

for i in range(7,9): 

    sns.countplot(f'nom_{i}', data= train, ax=ax[i-7],

                  order = train[f'nom_{i}'].dropna().value_counts().index)

    ax[i-7].set_ylim([0, 5500])

    ax[i-7].set_title(f'bin_{i}', fontsize=15)

    ax[i-7].set_xticks([])

fig.suptitle("Nominal Feature Distribution (Train Data)", fontsize=20)

plt.show()
fig, ax = plt.subplots(2,1, figsize=(30, 10))

for i in range(7,9): 

    sns.countplot(f'nom_{i}', hue='target', data= train, ax=ax[i-7],

                  order = train[f'nom_{i}'].dropna().value_counts().index)

    ax[i-7].set_ylim([0, 5000])

    ax[i-7].set_title(f'bin_{i}', fontsize=15)

    ax[i-7].set_xticks([])

fig.suptitle("Nominal Feature Distribution (Train Data)", fontsize=20)

plt.show()
train[[f'ord_{i}' for i in range(6)]].describe(include='all')
fig, ax = plt.subplots(1,3, figsize=(30, 8))



ord_order = [

    [1.0, 2.0, 3.0],

    ['Novice', 'Contributor', 'Expert', 'Master', 'Grandmaster'],

    ['Freezing', 'Cold', 'Warm', 'Hot', 'Boiling Hot', 'Lava Hot']

]



for i in range(3): 

    sns.countplot(f'ord_{i}', hue='target', data= train, ax=ax[i],

                  order = ord_order[i]

                 )

    ax[i].set_ylim([0, 200000])

    ax[i].set_title(f'ord_{i}', fontsize=15)

fig.suptitle("Ordinal Feature Distribution (Train Data)", fontsize=20)

plt.show()
fig, ax = plt.subplots(1,2, figsize=(24, 8))



for i in range(3, 5): 

    sns.countplot(f'ord_{i}', hue='target', data= train, ax=ax[i-3],

                  order = sorted(train[f'ord_{i}'].dropna().unique())

                 )

    ax[i-3].set_ylim([0, 75000])

    ax[i-3].set_title(f'ord_{i}', fontsize=15)

fig.suptitle("Ordinal Feature Distribution (Train Data 3~4)", fontsize=20)

plt.show()
for i in range(5):

    data = train[[f'ord_{i}', 'target']].groupby(f'ord_{i}')['target'].value_counts().unstack()

    data['rate'] = data[1]  / (data[0] + data[1] )

    data.sort_values(by=['rate'], inplace=True)

    display(data.style.highlight_max(color='lightgreen').highlight_min(color='#cd4f39'))
fig, ax = plt.subplots(2,1, figsize=(24, 16))



xlabels = train['ord_5'].dropna().value_counts().index



print(len(xlabels))



# just counting

sns.countplot('ord_5', data= train, ax=ax[0], order = xlabels )

ax[0].set_ylim([0, 12000])

ax[0].set_xticklabels(xlabels, rotation=90, rotation_mode="anchor", fontsize=7)



# with hue

sns.countplot('ord_5', hue='target', data= train, ax=ax[1], order = xlabels )

ax[1].set_ylim([0, 10000])

ax[1].set_xticklabels(xlabels, rotation=90, rotation_mode="anchor", fontsize=7)



fig.suptitle("Ordinal Feature Distribution (Train Data 5)", fontsize=20)

plt.show()
fig, ax = plt.subplots(2,1, figsize=(24, 16))



sns.countplot('day', hue='target', data= train, ax=ax[0])

ax[0].set_ylim([0, 100000])



sns.countplot('month', hue='target', data= train, ax=ax[1])

ax[1].set_ylim([0, 80000])



fig.suptitle("Day & Month Distribution", fontsize=20)

plt.show()
data = train[['day', 'target']].groupby('day')['target'].value_counts().unstack()

data['rate'] = data[1]  / (data[0] + data[1] )

data.sort_values(by=['rate'], inplace=True)

display(data.style.highlight_max(color='lightgreen').highlight_min(color='#cd4f39'))



data = train[['month', 'target']].groupby('month')['target'].value_counts().unstack()

data['rate'] = data[1]  / (data[0] + data[1] )

data.sort_values(by=['rate'], inplace=True)

display(data.style.highlight_max(color='lightgreen').highlight_min(color='#cd4f39'))
%%time

bin_encoding = {'F':0, 'T':1, 'N':0, 'Y':1}

train['bin_3'] = train['bin_3'].map(bin_encoding)

train['bin_4'] = train['bin_4'].map(bin_encoding)



test['bin_3'] = test['bin_3'].map(bin_encoding)

test['bin_4'] = test['bin_4'].map(bin_encoding)
%%time

from category_encoders.target_encoder import TargetEncoder



for i in range(10):

    label = TargetEncoder()

    train[f'nom_{i}'] = label.fit_transform(train[f'nom_{i}'].fillna('NULL'), target)

    test[f'nom_{i}'] = label.transform(test[f'nom_{i}'].fillna('NULL'))
%%time

ord_order = [

    [1.0, 2.0, 3.0],

    ['Novice', 'Contributor', 'Expert', 'Master', 'Grandmaster'],

    ['Freezing', 'Cold', 'Warm', 'Hot', 'Boiling Hot', 'Lava Hot']

]



for i in range(1, 3):

    ord_order_dict = {i : j for j, i in enumerate(ord_order[i])}

    train[f'ord_{i}'] = train[f'ord_{i}'].fillna('NULL').map(ord_order_dict)

    test[f'ord_{i}'] = test[f'ord_{i}'].fillna('NULL').map(ord_order_dict)
%%time

for i in range(3, 6):

    ord_order_dict = {i : j for j, i in enumerate(sorted(list(set(list(train[f'ord_{i}'].dropna().unique()) + list(test[f'ord_{i}'].dropna().unique())))))}

    train[f'ord_{i}'] = train[f'ord_{i}'].fillna('NULL').map(ord_order_dict)

    test[f'ord_{i}'] = test[f'ord_{i}'].fillna('NULL').map(ord_order_dict)
train.head()
%%time

f, ax = plt.subplots(1, 3, figsize=(45, 14))

for idx, tp in  enumerate(['pearson', 'kendall', 'spearman']) :

    corr = train.fillna(-1).corr(tp)

    mask = np.zeros_like(corr, dtype=np.bool)

    mask[np.triu_indices_from(mask)] = True

    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.2, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax[idx])

    ax[idx].set_title(f'{tp} correlation viz')

plt.show()
%%time

f, ax = plt.subplots(1, 3, figsize=(45, 14))

for idx, tp in  enumerate(['pearson', 'kendall', 'spearman']) :

    corr = test.fillna(-1).corr(tp)

    mask = np.zeros_like(corr, dtype=np.bool)

    mask[np.triu_indices_from(mask)] = True

    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.1, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax[idx])

    ax[idx].set_title(f'{tp} correlation viz (test)')

plt.show()