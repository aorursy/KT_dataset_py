import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('white')



df = pd.read_json('/kaggle/input/pokemon-gen-vii-pokedex/pokedex.json')

print(df.columns)

df.head(3)
pgrid = sns.PairGrid(df[['Total', 'Capture Rate', 'Egg Steps']], aspect=2, diag_sharey=False)

pgrid.fig.subplots_adjust(hspace=0.2, wspace=0.2)

pgrid.map_diag(plt.hist, bins='auto', histtype='stepfilled')

pgrid.map_offdiag(plt.scatter)

for ax in pgrid.axes.flat:

    ax.yaxis.set_tick_params(left=True, labelleft=True)

    ax.xaxis.set_tick_params(bottom=True, labelbottom=True)
print(df[df['Total']>=600]['Name'].values)
print(df[df['Total']>=550]['Name'].values)
print(np.sort(df['Capture Rate'].unique()),'\n')

print(df[df['Capture Rate']==3]['Name'].values, '\n')
sns.relplot(data=df, x='Total', y='Capture Rate', aspect=2)
from sklearn.neighbors import LocalOutlierFactor



clf = LocalOutlierFactor(n_neighbors=20, contamination='auto')

train_array = df[['Total','Capture Rate']]

pred = clf.fit_predict(train_array)
df['pred_LOF'] = pred

df[(df['pred_LOF']==-1)]['Name'].values
scores = clf.negative_outlier_factor_

df['score_LOF'] = scores

df[(df['score_LOF']<=-5)][['Name','score_LOF']]
df[(df['score_LOF']>=-5) & (df['score_LOF']<=-1.5)][['Name','score_LOF']]
from sklearn.ensemble import IsolationForest
clf = IsolationForest(behaviour='new', contamination='auto', random_state=24)

pred = clf.fit_predict(train_array)

df['pred_IF'] = pred

df['score_IF'] = clf.score_samples(train_array)

#df[(df['pred_IF']==-1) & (df['Total']>500)]['Name'].values



kde_bool = False

sns.distplot(df[(df['pred_IF']==-1)]['score_IF'], kde=kde_bool, color='red')

sns.distplot(df[(df['pred_IF']==1)]['score_IF'], kde=kde_bool, color='skyblue')
len(df[(df['pred_IF']==-1)]['Name'])
df[(df['pred_IF']==-1)]['Name'].values
df[(df['pred_IF']==-1) & (df['Total']>=550)]['Name'].values
print(len(df[(df['Total']>=550)]))

print(len(df[(df['pred_IF']==-1) & (df['Total']>=550)]))



series_varcut = df[(df['Total']>=550)]['Name']

series_IF = df[(df['pred_IF']==-1) & (df['Total']>=550)]['Name']

series_varcut[~series_varcut.isin(series_IF)]
clf = IsolationForest(behaviour='new', contamination=0.1, random_state=24)

pred = clf.fit_predict(train_array)

df['pred_IF_0.1'] = pred

df['score_IF_0.1'] = clf.score_samples(train_array)



sns.distplot(df[(df['pred_IF_0.1']==-1)]['score_IF_0.1'], kde=kde_bool, color='red')

sns.distplot(df[(df['pred_IF_0.1']==1)]['score_IF_0.1'], kde=kde_bool, color='skyblue')
df[(df['pred_IF_0.1']==-1)]['Name'].values
train_array = df[['Total','Capture Rate', 'Egg Steps']]

clf = IsolationForest(behaviour='new', contamination=0.1, random_state=24)



pred = clf.fit_predict(train_array)

df['pred_IF_3D'] = pred

df['score_IF_3D'] = clf.score_samples(train_array)



sns.distplot(df[(df['pred_IF_3D']==-1)]['score_IF_3D'], kde=kde_bool, color='red')

sns.distplot(df[(df['pred_IF_3D']==1)]['score_IF_3D'], kde=kde_bool, color='skyblue')
df[(df['pred_IF_3D']==-1) & (df['Total']>500)]['Name'].values