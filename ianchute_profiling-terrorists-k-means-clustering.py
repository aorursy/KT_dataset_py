import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics.cluster import silhouette_score
from sklearn.preprocessing import scale, robust_scale

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

plt.rcParams['figure.figsize'] = (12, 8)
sns.set_palette('coolwarm')
sns.set_color_codes('bright')
data = pd.read_csv('../input/globalterrorismdb_0617dist.csv', encoding='latin1')
# outlier removal - remove massive terrorist attacks
data = data[data['nkill'] <= 4].reset_index(drop=True)
data = data[data['nwound'] <= 7].reset_index(drop=True)
c = data.count().sort_values().drop([
    'eventid', 'country', 'iyear', 'natlty1', 'longitude', 'latitude', 'targsubtype1'])
_ = data[c[c > 100000].keys()].var().sort_values().plot.barh()
features = [
    'longitude',
    'latitude',
    
    'nwound',
    'nkill',
    
    'natlty1_txt',
    'targtype1_txt',
    'targsubtype1_txt',
    'weaptype1_txt',
    'attacktype1_txt',
]

X = pd.get_dummies(data[features])
X = X.T[X.var() > 0.05].T.fillna(0)
X = X.fillna(0)

print('Shape:', X.shape)
X.head()
scores = {}
for k in range(2, 11):
    print(k, end=', ')
    scores[k] = KMeans(n_clusters=k).fit(X).score(X)
_ = pd.Series(scores).plot.bar()
data['Cluster'] = KMeans(n_clusters=6).fit_predict(X) + 1
print('Silhouette Score:', silhouette_score(X, data['Cluster'], sample_size=10000) * 10000 // 1 / 100, '%')
names = data.groupby('Cluster')['region_txt'].describe()['top'].values
data['ClusterName'] = data['Cluster'].apply(lambda c: names[c - 1])

numerical = data.dtypes[data.dtypes != 'object'].keys()
exclude = [
    'eventid', 'Cluster', 'region', 'country', 'iyear', 
    'natlty1', 'natlty2', 'natlty3', 'imonth', 'iday',
    'guncertain1', 'guncertain2', 'guncertain3'
] + [col for col in numerical if 'type' in col or 'mode' in col or 'ransom' in col]
X_profiling = data[numerical.drop(exclude)].fillna(0)
X_profiling = pd.DataFrame(scale(X_profiling), columns=X_profiling.columns)
X_profiling['ClusterName'] = data['ClusterName']
_ = sns.heatmap(X_profiling.groupby('ClusterName').mean().drop(['longitude', 'latitude'], axis=1).T, 
               cmap='coolwarm')
ckeys = data['ClusterName'].unique()
ckeys = dict(zip(ckeys, plt.cm.tab10(range(len(ckeys)))))

for i, x in X_profiling.groupby('ClusterName'):
    _ = plt.scatter(x['longitude'], x['latitude'], c=ckeys[i], marker='.', cmap='tab10', label=i)
_ = plt.legend(loc=3)
ckeys = data['region_txt'].unique()
ckeys = dict(zip(ckeys, plt.cm.tab20(range(len(ckeys)))))

for i, x in pd.concat([X_profiling, data['region_txt']], axis=1).groupby('region_txt'):
    _ = plt.scatter(x['longitude'], x['latitude'], c=ckeys[i], marker='.', cmap='tab10', label=i)
_ = plt.legend(loc=3)
print('Similarity between cluster and region labels:', 
      len(data[data['region_txt'] == data['ClusterName']]) / len(data) * 10000 // 1 / 100, '%')
d = pd.get_dummies(data['attacktype1_txt'])
d['ClusterName'] = data['ClusterName']
_ = sns.heatmap(d.groupby('ClusterName').mean().T, cmap='coolwarm')
d = pd.get_dummies(data['targtype1_txt'])
d['ClusterName'] = data['ClusterName']
_ = sns.heatmap(d.groupby('ClusterName').mean().T, cmap='coolwarm')
top = data['targsubtype1_txt'].value_counts().head(20).keys().tolist()
d = pd.get_dummies(data['targsubtype1_txt'].apply(lambda x: x if x in top else None))
d['ClusterName'] = data['ClusterName']
_ = sns.heatmap(d.groupby('ClusterName').mean().T, cmap='coolwarm')
d = pd.get_dummies(data['weaptype1_txt'])
d['ClusterName'] = data['ClusterName']
_ = sns.heatmap(d.groupby('ClusterName').mean().T, cmap='coolwarm')
top_natls = data['natlty1_txt'].value_counts().head(20).keys()
natl = data['natlty1_txt'].apply(lambda x: x if x in top_natls else None)
d = pd.get_dummies(natl)
d['ClusterName'] = data['ClusterName']
_ = sns.heatmap(d.groupby('ClusterName').mean().T, cmap='coolwarm')
months = ['', 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
d = pd.get_dummies(data['imonth'].apply(lambda x: None if x == 0 else months[int(x)]))
d['ClusterName'] = data['ClusterName']
_ = sns.heatmap(d.groupby('ClusterName').mean().T, cmap='coolwarm')
d = pd.get_dummies(data['iday'].apply(lambda x: None if x == 0 else int(x)))
d['ClusterName'] = data['ClusterName']
_ = sns.heatmap(d.groupby('ClusterName').mean().T, cmap='coolwarm')
d = pd.get_dummies(data['nhours'].apply(lambda x: None if x <= 0 else x))
d['ClusterName'] = data['ClusterName']
_ = sns.heatmap(d.groupby('ClusterName').mean().T, cmap='coolwarm')
d = pd.get_dummies(data['nperps'].apply(lambda x: None if (x <= 0 or x >= 20) else x))
d['ClusterName'] = data['ClusterName']
_ = sns.heatmap(d.groupby('ClusterName').mean().T, cmap='coolwarm')