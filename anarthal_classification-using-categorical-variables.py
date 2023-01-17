import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from matplotlib import pyplot as plt
import pylab
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import xgboost

pylab.rcParams['figure.figsize'] = (15.0, 8.0)
sns.set()
SEED = 42
def mapord(s, values):
    vmap = {name: i for i, name in enumerate(values)}
    return s.map(lambda x: vmap.get(x, x))
def mapletter(s):
    return s.map(lambda x: ord(x.lower()) - ord('a') if type(x) is str else x)
def split(X, y):
    return train_test_split(X, y, random_state=SEED, stratify=y)
def proportion_plot(df, x, cat, kind='bar', **kwargs):
    df.groupby(x)[cat].value_counts(normalize=True)\
    .rename('count')\
    .reset_index()\
    .pivot(index=x, columns=cat, values='count')\
    .plot(kind=kind, stacked=True, **kwargs)
dftrain = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/train.csv')
dftest = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/test.csv')
sample = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/sample_submission.csv')
dftrain.set_index('id', inplace=True)
dftest.set_index('id', inplace=True)
proportion_plot(dftrain, 'day', 'target')
proportion_plot(dftrain, 'month', 'target')
fig, ax = plt.subplots(2,2, figsize=(17, 10))
sns.countplot(data=dftrain, x='bin_0', ax=ax[0][0])
sns.countplot(data=dftrain, x='bin_1', ax=ax[0][1])
sns.countplot(data=dftrain, x='bin_2', ax=ax[1][0])
sns.countplot(data=dftrain, x='bin_3', ax=ax[1][1])
fig, ax = plt.subplots(3,2, figsize=(17, 20))
sns.countplot(data=dftrain, x='nom_0', ax=ax[0][0])
sns.countplot(data=dftrain, x='nom_1', ax=ax[0][1])
sns.countplot(data=dftrain, x='nom_2', ax=ax[1][0])
sns.countplot(data=dftrain, x='nom_3', ax=ax[1][1])
sns.countplot(data=dftrain, x='nom_4', ax=ax[2][0])
fig, ax = plt.subplots(3,2, figsize=(17, 20))
sns.countplot(data=dftrain, x='ord_0', ax=ax[0][0])
sns.countplot(data=dftrain, x='ord_1', ax=ax[0][1])
sns.countplot(data=dftrain, x='ord_2', ax=ax[1][0])
sns.countplot(data=dftrain, x='ord_3', ax=ax[1][1])
sns.countplot(data=dftrain, x='ord_4', ax=ax[2][0])
dftrain.fillna()
def fillna_mode(s):
    return s.fillna(s.mode().iloc[0])

def map2letters(s):
    values = sorted(s.unique())
    dct = { v: i for i, v in enumerate(values) }
    return s.map(lambda x: dct[x])
    

# Encoding and imputation
y = dftrain['target'].copy()
dfall = pd.concat([dftrain.drop(columns='target'), dftest])
drop = []

# Binary
dfall['bin_0'] = fillna_mode(dfall['bin_0'])
dfall['bin_1'] = fillna_mode(dfall['bin_1'])
dfall['bin_2'] = fillna_mode(dfall['bin_2'])
tf_map = {'T': 1.0, 'F': 0.0}
dfall['bin_3'] = fillna_mode(dfall['bin_3']).map(lambda x: tf_map[x])
yn_map = {'Y': 1.0, 'N': 0.0}
dfall['bin_4'] = fillna_mode(dfall['bin_4']).map(lambda x: yn_map[x])

# Nominal
num_hot_encodes = 4
hot_encodes = ['nom_{}'.format(i) for i in range(0, num_hot_encodes+1)]
for name in hot_encodes: # TODO: reduce cardinality for the other ords
    dfall[name] = fillna_mode(dfall[name])
dfall = pd.get_dummies(dfall, columns=hot_encodes, drop_first=True)
drop += ['nom_{}'.format(i) for i in range(num_hot_encodes+1, 10)]

# Ordinal
dfall['ord_0'] = fillna_mode(dfall['ord_0'])
dfall['ord_1'] = mapord(fillna_mode(dfall['ord_1']), ['Novice', 'Contributor', 'Expert', 'Master', 'Grandmaster'])
dfall['ord_2'] = mapord(fillna_mode(dfall['ord_2']), ['Freezing', 'Cold', 'Warm', 'Hot', 'Boiling Hot', 'Lava Hot'])
dfall['ord_3'] = mapletter(fillna_mode(dfall['ord_3']))
dfall['ord_4'] = mapletter(fillna_mode(dfall['ord_4']))
dfall['ord_5'] = map2letters(fillna_mode(dfall['ord_5']))

# Day: days 1, 2, 6, 7 are similar; 3, 4, 5 too. Month: as is
dfall['day_extreme'] = dfall['day'].isin([1.0, 2.0, 6.0, 7.0]).astype(float)

# Month
drop.append('day')

# Drop
dfall.drop(columns=drop, inplace=True)

# Setup for modelling
X = dfall.loc[y.index, :].copy()
X_target = dfall.loc[dftest.index, :]
X_train, X_test, y_train, y_test = split(X, y)
fig, ax = plt.subplots(3,2, figsize=(17, 20))
for i in range(0, 5):
    proportion_plot(X.assign(target=y), 'ord_{}'.format(i), 'target', ax=ax.flatten()[i])
X.assign(target=y).groupby('ord_5')['target'].value_counts(normalize=True)\
    .rename('count')\
    .reset_index()\
    .pivot(index='ord_5', columns='target', values='count')[1].plot.line()
corr = X.assign(target=y).corr()
idx = corr['target'].abs().sort_values(ascending=False).index
sns.heatmap(corr.loc[idx, idx], cmap=plt.cm.BrBG)
model = xgboost.XGBClassifier(random_state=SEED, objective='binary:logistic')
model.fit(X_train, y_train)
model.score(X_test, y_test)
probs = model.predict_proba(X_test)
roc_auc_score(y_test, probs[:, 1])
model.fit(X, y)
pred = model.predict_proba(X_target)
dfout = pd.DataFrame({'id': X_target.index, 'target': pred[:, 1]})
dfout.to_csv('awesome.csv', index=False)
