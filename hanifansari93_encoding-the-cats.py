import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/train.csv')

test = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/test.csv')
# Number of nulls per each column

nulls = train.isna().sum().sort_values(ascending = False)[train.isna().sum() != 0]

perc = round(train.isna().sum()/len(train) * 100,2)[train.isna().sum()/len(train) != 0]

pd.concat([nulls,perc], axis = 1, sort = False, keys=['Total','Percent'])
for column in train.columns:

    train[column].fillna(train[column].mode()[0], inplace=True)

for column in test.columns:

    test[column].fillna(test[column].mode()[0], inplace=True)
#Plotting binary columns

cols = ['bin_0','bin_1','bin_2','bin_3','bin_4']

fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(16,8), dpi= 60)

axes = axes.flatten()

for i, ax in zip(cols, axes):

    sns.countplot(x = i, ax = ax, data = train, palette = 'Paired')

    total = len(train[i])

    for p in ax.patches:

        percentage = '{:.1f}%'.format(100 * p.get_height()/total)

        x = p.get_x() + p.get_width() - 0.5

        y = p.get_y() + p.get_height() + 1000

        ax.annotate(percentage, (x, y))

        ax.set_ylabel('Count')

fig.delaxes(axes[5])

plt.tight_layout()
#Plotting nominal columns

cols = ['nom_0','nom_1','nom_2','nom_3','nom_4']

fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(18,8), dpi= 60)

axes = axes.flatten()

for i, ax in zip(cols, axes):

    sns.countplot(x = i, ax = ax, data = train, palette = 'Paired')

    total = len(train[i])

    for p in ax.patches:

        percentage = '{:.1f}%'.format(100 * p.get_height()/total)

        x = p.get_x() + p.get_width() - 0.5

        y = p.get_y() + p.get_height() + 1000

        ax.annotate(percentage, (x, y))

        ax.set_ylabel('Count')

fig.delaxes(axes[5])

plt.tight_layout()
cols = ['nom_5','nom_6','nom_7','nom_8','nom_9']

for col in cols:

    print (train[col].nunique())
#Plotting ordinal columns

cols = ['ord_0','ord_1','ord_2']

fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(18,4), dpi= 60)

axes = axes.flatten()

for i, ax in zip(cols, axes):

    sns.countplot(x = i, ax = ax, data = train, palette = 'Paired')

    total = len(train[i])

    for p in ax.patches:

        percentage = '{:.1f}%'.format(100 * p.get_height()/total)

        x = p.get_x() + p.get_width() - 0.5

        y = p.get_y() + p.get_height() + 1000

        ax.annotate(percentage, (x, y))

        ax.set_ylabel('Count')

plt.tight_layout()
cols = ['ord_3','ord_4','ord_5']

for col in cols:

    print (train[col].nunique())
from sklearn.model_selection import StratifiedKFold

import category_encoders as ce



cols = ['nom_5','nom_6','nom_7','nom_8','nom_9']

teTrain = train[cols]; teTest = test[cols]

train_y = train['target']; test_id = test['id']

col_to_encode = teTrain.columns.tolist()

oof = pd.DataFrame([])



for tr_idx, oof_idx in StratifiedKFold(n_splits=5, random_state=42, shuffle=True).split(teTrain, train_y):

    ce_target_encoder = ce.TargetEncoder(cols = col_to_encode, smoothing=0.2)

    ce_target_encoder.fit(teTrain.iloc[tr_idx, :], train_y.iloc[tr_idx])

    oof = oof.append(ce_target_encoder.transform(teTrain.iloc[oof_idx, :]), ignore_index=False)

ce_target_encoder = ce.TargetEncoder(cols = col_to_encode, smoothing=0.2)

ce_target_encoder.fit(teTrain, train_y)

teTrain = oof.sort_index()

teTest = ce_target_encoder.transform(teTest)



train[cols] = teTrain[cols]

test[cols] = teTest[cols]
map_ord_1 = {'Novice':1, 'Contributor':2, 'Expert':3, 'Master':4, 'Grandmaster':5}

map_ord_2 = {'Freezing': 1, 'Cold':2, 'Warm':3, 'Hot':4, 'Boiling Hot': 5, 'Lava Hot':6}

map_ord_3 = dict(zip(train['ord_3'].value_counts().sort_index().keys(),range(1, len(train['ord_3'].value_counts())+1)))

map_ord_4 = dict(zip(train['ord_4'].value_counts().sort_index().keys(),range(1, len(train['ord_4'].value_counts())+1)))



temp_ord_5 = pd.DataFrame(train['ord_5'].value_counts().sort_index().keys(), columns=['ord_5'])

temp_ord_5['First'] = temp_ord_5['ord_5'].astype(str).str[0].str.upper()

temp_ord_5['Second'] = temp_ord_5['ord_5'].astype(str).str[1].str.upper()

temp_ord_5['First'] = temp_ord_5['First'].replace(map_ord_4)

temp_ord_5['Second'] = temp_ord_5['Second'].replace(map_ord_4)

temp_ord_5['Add'] = temp_ord_5['First']+temp_ord_5['Second']

temp_ord_5['Mul'] = temp_ord_5['First']*temp_ord_5['Second']

map_ord_5 = dict(zip(temp_ord_5['ord_5'],temp_ord_5['Mul']))



train['ord_1'] = train['ord_1'].replace(map_ord_1)

train['ord_2'] = train['ord_2'].replace(map_ord_2)

train['ord_3'] = train['ord_3'].replace(map_ord_3)

train['ord_4'] = train['ord_4'].replace(map_ord_4)

train['ord_5'] = train['ord_5'].replace(map_ord_5)



test['ord_1'] = test['ord_1'].replace(map_ord_1)

test['ord_2'] = test['ord_2'].replace(map_ord_2)

test['ord_3'] = test['ord_3'].replace(map_ord_3)

test['ord_4'] = test['ord_4'].replace(map_ord_4)

test['ord_5'] = test['ord_5'].replace(map_ord_5)
map_bin_3 = {'F':0, 'T':1}

map_bin_4 = {'N': 0, 'Y':1}



train['bin_3'] = train['bin_3'].replace(map_bin_3)

train['bin_4'] = train['bin_4'].replace(map_bin_4)



test['bin_3'] = test['bin_3'].replace(map_bin_3)

test['bin_4'] = test['bin_4'].replace(map_bin_4)
cols = ['nom_0','nom_1','nom_2','nom_3','nom_4']

train = pd.concat((train, pd.get_dummies(train[cols], drop_first=True)),1)

train = train.drop(cols, axis = 1)

test = pd.concat((test, pd.get_dummies(test[cols], drop_first=True)),1)

test = test.drop(cols, axis = 1)
cols = ['day', 'month']

for feature in cols:

    train[feature+'_sin'] = np.sin((2*np.pi*train[feature])/max(train[feature]))

    train[feature+'_cos'] = np.cos((2*np.pi*train[feature])/max(train[feature]))

    test[feature+'_sin'] = np.sin((2*np.pi*test[feature])/max(test[feature]))

    test[feature+'_cos'] = np.cos((2*np.pi*test[feature])/max(test[feature]))

train = train.drop(cols, axis=1)

test = test.drop(cols, axis=1)