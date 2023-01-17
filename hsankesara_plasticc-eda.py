# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
DATA_PATH='../input/'
df_train = pd.read_csv(DATA_PATH + 'training_set.csv')
df_train_meta = pd.read_csv(DATA_PATH + 'training_set_metadata.csv')
df_train.head()
df_train.describe()
df_train.info()
df_train_meta.head()
df_train_meta.describe()
df_train_meta.info()
f, ax = plt.subplots(figsize=(12, 6))
sns.distplot(df_train['flux'], ax=ax)
plt.show()
df_train.flux.describe()
f, ax = plt.subplots(figsize=(12, 6))
sns.distplot(df_train['flux_err'], ax=ax)
plt.show()
df_train.flux_err.describe()
df_train[(df_train.flux> 1000) | (df_train.flux < -1000)].describe()
sns.distplot(df_train[(df_train.flux < 1000) & (df_train.flux > -1000)].flux)
sns.countplot(df_train[(df_train.flux < 1000) & (df_train.flux > -1000)].detected)
sns.countplot(df_train[(df_train.flux> 250) | (df_train.flux < -250)].detected)
sns.distplot(df_train[(df_train.flux < 100) & (df_train.flux > -100)].flux)
sns.countplot(df_train[(df_train.flux_err >= 100) | (df_train.flux_err <= 100)].detected)
sns.countplot(df_train[(df_train.flux_err > 100) | (df_train.flux_err < -100)].detected)
sns.heatmap(df_train[['flux', 'flux_err']].corr(), annot=True)

passbands = [0, 1,2,3,4,5]
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(18,10))
i = 0
for row in ax:
    for col in row:
        sns.distplot(df_train[(df_train.passband == passbands[i]) & (df_train.flux < 250) & (df_train.flux > -250)]['flux'], ax=col, axlabel='flux distribution of passband ' + str(i))
        i += 1
plt.show()
        
sns.countplot(x='detected', data=df_train, hue='passband')
plt.show()
df_train.groupby(['passband']).count()
ts_lens = df_train.groupby(['object_id', 'passband']).size()
f, ax = plt.subplots(figsize=(12, 6))
sns.distplot(ts_lens, ax=ax)
ax.set_title('distribution of time series lengths')
plt.show()
passbands = [0, 1,2,3,4,5]
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(18,10))
i = 0
for row in ax:
    for col in row:
        sns.distplot(df_train[df_train.passband == i].groupby(['object_id']).size(), ax=col, axlabel='timeseries distribution of passband ' + str(i))
        i += 1
plt.show()
f, ax = plt.subplots(figsize=(12, 6))
sns.distplot(df_train['mjd'], ax=ax, bins=200)
ax.set_title('number of observations made at each time point')
plt.show()
f, ax = plt.subplots(figsize=(12, 6))
sns.distplot(df_train[df_train['object_id'] == 713]['mjd'], ax=ax, bins=200)
ax.set_title('number of observations made at each time point')
plt.show()
passbands = [0, 1,2,3,4,5]
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(22,10))
i = 0
for row in ax:
    for col in row:
        sns.distplot(df_train[(df_train['object_id'] == 713) & (df_train['passband'] == 3)]['mjd'], ax=col, bins=200)
        col.set_title('number of observations made at each time point by passband ' +  str(i))
        i += 1
plt.show()

f, ax = plt.subplots(figsize=(12, 9))
ax.scatter(x='mjd', y='flux', data=df_train.groupby(['mjd']).mean().reset_index())
ax.scatter(x='mjd', y='flux_err', data=df_train.groupby(['mjd']).mean().reset_index())
ax.legend(['flux', 'flux error'])
plt.show()
f, ax = plt.subplots(figsize=(12, 9))
ax.scatter(x='mjd', y='flux', data=df_train[df_train.object_id==713].groupby(['mjd']).mean().reset_index())
ax.scatter(x='mjd', y='flux_err', data=df_train[df_train.object_id==713].groupby(['mjd']).mean().reset_index())
ax.legend(['flux', 'flux error'])
plt.show()
f, ax = plt.subplots(figsize=(12, 9))
ax.scatter(x='mjd', y='flux', data=df_train[df_train.object_id==615].groupby(['mjd']).mean().reset_index())
ax.scatter(x='mjd', y='flux_err', data=df_train[df_train.object_id==615].groupby(['mjd']).mean().reset_index())
ax.legend(['flux', 'flux error'])
plt.show()
objects = df_train.object_id.unique()
random_id = np.random.randint(0, len(objects), 12)
fig, ax = plt.subplots(nrows=4, ncols=3, figsize=(18,10))
i = 0
for row in ax:
    for col in row:
        col.scatter(x='mjd', y='flux', data=df_train[df_train.object_id==objects[random_id[i]]].groupby(['mjd']).mean().reset_index())
        col.scatter(x='mjd', y='flux_err', data=df_train[df_train.object_id==objects[random_id[i]]].groupby(['mjd']).mean().reset_index())
        col.legend(['flux', 'flux error'])
        i += 1
plt.show()
sns.heatmap(df_train[['mjd', 'detected']].corr(), annot=True)
plt.show()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(df_train_meta[['ra', 'decl', 'gal_l', 'gal_b']].corr(), annot=True, ax=ax)
plt.show()
f, ax = plt.subplots(figsize=(21, 9))
ax.scatter(df_train_meta.object_id, df_train_meta.hostgal_specz)
ax.scatter(df_train_meta.object_id, df_train_meta.hostgal_photoz)
ax.legend(['specz redshift', 'photoz redshift'])
plt.show()
f, ax = plt.subplots(figsize=(21, 9))
ax.scatter(df_train_meta.object_id, df_train_meta.hostgal_photoz_err)
ax.scatter(df_train_meta.object_id, df_train_meta.hostgal_photoz)
ax.legend(['photoz redshift error', 'photoz redshift'])
plt.show()
f, ax = plt.subplots(figsize=(12, 9))
sns.distplot(df_train_meta.hostgal_photoz)
sns.distplot(df_train_meta.hostgal_specz)
ax.legend(['photoz redshift', 'specz redshift'])
plt.show()
f, ax = plt.subplots(figsize=(21, 9))
sns.distplot(df_train_meta.hostgal_photoz)
sns.distplot(df_train_meta.hostgal_photoz_err)
ax.legend(['photoz redshift',  'photoz redshift error'])
plt.show()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(df_train_meta[['hostgal_specz', 'hostgal_photoz', 'hostgal_photoz_err']].corr(), annot=True, ax=ax)
plt.show()
sns.countplot(df_train_meta.target)
plt.show()
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
sns.countplot(df_train_meta[df_train_meta['hostgal_photoz'] == 0].target, ax = ax[0])
sns.countplot(df_train_meta[df_train_meta['hostgal_photoz'] != 0].target, ax = ax[1])
plt.show()
f, ax = plt.subplots(figsize=(12, 9))
sns.countplot('target', hue='ddf', data=df_train_meta, ax=ax)
plt.show()
f, ax = plt.subplots(figsize=(12, 6))
sns.distplot(df_train_meta[df_train_meta.ddf==1].hostgal_photoz)
sns.distplot(df_train_meta[df_train_meta.ddf==0].hostgal_photoz)
ax.legend(['Redshift on DDF Survey Area',  'Redshift outside DDF survey Area'])
plt.show()
f, ax = plt.subplots(figsize=(12, 6))
sns.distplot(df_train_meta[df_train_meta.ddf==1].hostgal_photoz_err)
sns.distplot(df_train_meta[df_train_meta.ddf==0].hostgal_photoz_err)
ax.legend(['Redshift error on DDF Survey Area',  'Redshift error outside DDF survey Area'])
plt.show()
f, ax = plt.subplots(figsize=(12, 6))
sns.distplot(df_train_meta.hostgal_photoz)
sns.distplot(df_train_meta.mwebv)
ax.legend(['photoz redshift',  'MWEBV'])
plt.show()
f, ax = plt.subplots(figsize=(12, 6))
sns.distplot(df_train_meta[df_train_meta.hostgal_photoz==0].mwebv)
sns.distplot(df_train_meta[df_train_meta.hostgal_photoz!=0].mwebv)
ax.legend(['Galactic MWEBV',  'Extragalactic MWEBV'])
plt.show()
sns.distplot(df_train_meta.distmod.dropna())
plt.show()
f, ax = plt.subplots(figsize=(12, 6))
sns.distplot(df_train_meta[df_train_meta.hostgal_photoz == 0].dropna().distmod)
sns.distplot(df_train_meta[df_train_meta.hostgal_photoz != 0].dropna().distmod)
ax.legend(['Galactic distmod',  'Extragalactic distmod'])
plt.show()
print(df_train_meta[df_train_meta.hostgal_photoz != 0].info())
print(df_train_meta[df_train_meta.hostgal_photoz == 0].info())
f, ax = plt.subplots(figsize=(12, 6))
sns.violinplot(y='distmod', x='target', data=df_train_meta, ax=ax)
plt.show()




