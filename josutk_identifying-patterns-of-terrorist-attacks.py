# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
!pip3 install pyod  
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
df_terrorism = pd.read_csv('/kaggle/input/gtd/globalterrorismdb_0718dist.csv', encoding = "ISO-8859-1")
df_terrorism = df_terrorism[['iyear','imonth','iday','extended','country','country_txt','region', 'region_txt','success',  'attacktype1_txt', 'targtype1', 'targtype1_txt', 'gname', 'weaptype1_txt']]
df_terrorism.head()
df_terrorism.head()
count = df_terrorism.pivot_table(columns='gname', aggfunc='size', fill_value=0)
terror_gname = dict(zip(count.index, count[:]))
terror_gname = sorted(terror_gname.items(), key=lambda kv: kv[1], reverse=True)
terror_gname = dict(terror_gname)
terror_gname_11_keys = list(terror_gname.keys())
terror_gname_10_values = list(terror_gname.values())
terror_gname_10_values = terror_gname_10_values[:10]
names =terror_gname_11_keys[0:11]
countries = pd.get_dummies(df_terrorism['country_txt'])
countries.reset_index(drop=True, inplace=True)
regions = pd.get_dummies(df_terrorism['region_txt'])
regions.reset_index(drop=True, inplace=True)
attack = pd.get_dummies(df_terrorism['attacktype1_txt'])
attack.reset_index(drop=True, inplace=True)
target = pd.get_dummies(df_terrorism['targtype1_txt'])
target.reset_index(drop=True, inplace=True)
weapon = pd.get_dummies(df_terrorism['weaptype1_txt'])
weapon.reset_index(drop=True, inplace=True)
df_terrorism.reset_index(drop=True, inplace=True)
df_terrorism_new = pd.concat([df_terrorism, countries, regions, attack, target, weapon], axis=1)
df_terrorism_new = df_terrorism_new.drop(['iyear','imonth','iday','country_txt', 'region_txt','attacktype1_txt','targtype1_txt','weaptype1_txt'] ,axis=1)

from pyod.models.ocsvm import OCSVM
from tqdm import tqdm
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)

def get_indexs(name, df):
    tmp = df[df['gname']==name]
    X_tmp = tmp.iloc[:,~tmp.columns.isin(['gname'])]
    clf = OCSVM(gamma='auto').fit(X_tmp)
    tmp['outlier'] = clf.labels_
    indexs_out  = tmp[tmp['outlier']==1].index
    indexs_in = tmp[tmp['outlier']==0].index
    return indexs_out, indexs_in

indexs_out, indexs_in = get_indexs(names[1], df_terrorism_new)
df_index_in = df_terrorism[df_terrorism.index.isin(indexs_in)]
df_index_out = df_terrorism[df_terrorism.index.isin(indexs_out)]

df_index_out['outlier'] = 'nonstandard'
df_index_in['outlier'] = 'standard'

select_group =pd.concat([df_index_in, df_index_out])

fig = px.sunburst(select_group, path=['outlier','country_txt','attacktype1_txt','targtype1_txt','weaptype1_txt'])
fig.update_layout(
    #grid= dict(columns=300, rows=300),
    margin = dict(t=0, l=0, r=0, b=0)
)
fig.show()
indexs_out, indexs_in = get_indexs(names[2], df_terrorism_new)
df_index_in = df_terrorism[df_terrorism.index.isin(indexs_in)]
df_index_out = df_terrorism[df_terrorism.index.isin(indexs_out)]

df_index_out['outlier'] = 'nonstandard'
df_index_in['outlier'] = 'standard'

select_group =pd.concat([df_index_in, df_index_out])

fig = px.sunburst(select_group, path=['outlier','country_txt','attacktype1_txt','targtype1_txt','weaptype1_txt'])
fig.update_layout(
    #grid= dict(columns=300, rows=300),
    margin = dict(t=0, l=0, r=0, b=0)
)
fig.show()
indexs_out, indexs_in = get_indexs(names[3], df_terrorism_new)
df_index_in = df_terrorism[df_terrorism.index.isin(indexs_in)]
df_index_out = df_terrorism[df_terrorism.index.isin(indexs_out)]

df_index_out['outlier'] = 'nonstandard'
df_index_in['outlier'] = 'standard'

select_group =pd.concat([df_index_in, df_index_out])

fig = px.sunburst(select_group, path=['outlier','country_txt','attacktype1_txt','targtype1_txt','weaptype1_txt'])
fig.update_layout(
    #grid= dict(columns=300, rows=300),
    margin = dict(t=0, l=0, r=0, b=0)
)
fig.show()
indexs_out, indexs_in = get_indexs(names[4], df_terrorism_new)
df_index_in = df_terrorism[df_terrorism.index.isin(indexs_in)]
df_index_out = df_terrorism[df_terrorism.index.isin(indexs_out)]

df_index_out['outlier'] = 'nonstandard'
df_index_in['outlier'] = 'standard'

select_group =pd.concat([df_index_in, df_index_out])

fig = px.sunburst(select_group, path=['outlier','country_txt','attacktype1_txt','targtype1_txt','weaptype1_txt'])
fig.update_layout(
    #grid= dict(columns=300, rows=300),
    margin = dict(t=0, l=0, r=0, b=0)
)
fig.show()
indexs_out, indexs_in = get_indexs(names[5], df_terrorism_new)
df_index_in = df_terrorism[df_terrorism.index.isin(indexs_in)]
df_index_out = df_terrorism[df_terrorism.index.isin(indexs_out)]

df_index_out['outlier'] = 'nonstandard'
df_index_in['outlier'] = 'standard'

select_group =pd.concat([df_index_in, df_index_out])

fig = px.sunburst(select_group, path=['outlier','country_txt','attacktype1_txt','targtype1_txt','weaptype1_txt'])
fig.update_layout(
    #grid= dict(columns=300, rows=300),
    margin = dict(t=0, l=0, r=0, b=0)
)
fig.show()
indexs_out, indexs_in = get_indexs(names[6], df_terrorism_new)
df_index_in = df_terrorism[df_terrorism.index.isin(indexs_in)]
df_index_out = df_terrorism[df_terrorism.index.isin(indexs_out)]

df_index_out['outlier'] = 'nonstandard'
df_index_in['outlier'] = 'standard'

select_group =pd.concat([df_index_in, df_index_out])

fig = px.sunburst(select_group, path=['outlier','country_txt','attacktype1_txt','targtype1_txt','weaptype1_txt'])
fig.update_layout(
    #grid= dict(columns=300, rows=300),
    margin = dict(t=0, l=0, r=0, b=0)
)
fig.show()
indexs_out, indexs_in = get_indexs(names[7], df_terrorism_new)
df_index_in = df_terrorism[df_terrorism.index.isin(indexs_in)]
df_index_out = df_terrorism[df_terrorism.index.isin(indexs_out)]

df_index_out['outlier'] = 'nonstandard'
df_index_in['outlier'] = 'standard'

select_group =pd.concat([df_index_in, df_index_out])

fig = px.sunburst(select_group, path=['outlier','country_txt','attacktype1_txt','targtype1_txt','weaptype1_txt'])
fig.update_layout(
    #grid= dict(columns=300, rows=300),
    margin = dict(t=0, l=0, r=0, b=0)
)
fig.show()
indexs_out, indexs_in = get_indexs(names[8], df_terrorism_new)
df_index_in = df_terrorism[df_terrorism.index.isin(indexs_in)]
df_index_out = df_terrorism[df_terrorism.index.isin(indexs_out)]

df_index_out['outlier'] = 'nonstandard'
df_index_in['outlier'] = 'standard'

select_group =pd.concat([df_index_in, df_index_out])

fig = px.sunburst(select_group, path=['outlier','country_txt','attacktype1_txt','targtype1_txt','weaptype1_txt'])
fig.update_layout(
    #grid= dict(columns=300, rows=300),
    margin = dict(t=0, l=0, r=0, b=0)
)
fig.show()
indexs_out, indexs_in = get_indexs(names[9], df_terrorism_new)
df_index_in = df_terrorism[df_terrorism.index.isin(indexs_in)]
df_index_out = df_terrorism[df_terrorism.index.isin(indexs_out)]

df_index_out['outlier'] = 'nonstandard'
df_index_in['outlier'] = 'standard'

select_group =pd.concat([df_index_in, df_index_out])

fig = px.sunburst(select_group, path=['outlier','country_txt','attacktype1_txt','targtype1_txt','weaptype1_txt'])
fig.update_layout(
    #grid= dict(columns=300, rows=300),
    margin = dict(t=0, l=0, r=0, b=0)
)
fig.show()
indexs_out, indexs_in = get_indexs(names[10], df_terrorism_new)
df_index_in = df_terrorism[df_terrorism.index.isin(indexs_in)]
df_index_out = df_terrorism[df_terrorism.index.isin(indexs_out)]

df_index_out['outlier'] = 'nonstandard'
df_index_in['outlier'] = 'standard'

select_group =pd.concat([df_index_in, df_index_out])

fig = px.sunburst(select_group, path=['outlier','country_txt','attacktype1_txt','targtype1_txt','weaptype1_txt'])
fig.update_layout(
    #grid= dict(columns=300, rows=300),
    margin = dict(t=0, l=0, r=0, b=0)
)
fig.show()
unknown = df_terrorism[df_terrorism['gname'] =='Unknown']
unknown.head()
in_ = []
out_ = []

for i in range(1, 11): 
    indexs_out, indexs_in = get_indexs(names[i], df_terrorism_new)
    for index in indexs_in:
        in_.append(index)
    for index in indexs_out:
        out_.append(index)
pattern = df_terrorism_new[df_terrorism_new.index.isin(in_)]
from sklearn.cluster import KMeans
X = pattern.iloc[:,~pattern.columns.isin(['gname'])]
kmeans = KMeans(n_clusters=10, random_state=42).fit(X)
kmeans.labels_
import seaborn as sns
pattern['cluster'] = kmeans.labels_
pattern['cluster'].value_counts()
ax = sns.countplot(pattern['cluster'])
ax
ax = sns.countplot(pattern['gname'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)
pattern['x'] =  X_pca[:,0]
pattern['y'] = X_pca[:,1]
pattern['z'] = X_pca[:,2]
import plotly.express as px
fig = px.scatter_3d(pattern, x='x', y='y', z='z',
              color='cluster')
fig.show()
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

fig = make_subplots(rows=2, cols=5, specs=[[{"type": "pie"}, {"type": "pie"}, {"type": "pie"}, {"type": "pie"}, {"type": "pie"}],
                                          [{"type": "pie"}, {"type": "pie"}, {"type": "pie"}, {"type": "pie"}, {"type": "pie"}]])


j=1
k=1
for i in range(1, 11): 
    if ((i-1) %5 ==0) and (i !=1): 
        k = 1
        j+=1
    tmp = pattern[pattern['gname']==names[i]]
    fig.append_trace(go.Pie(values=list(tmp['cluster'].value_counts()), 
                            labels=tmp['cluster'].value_counts().index, title_text='Clustering Distribuition '+str(names[i])), row=j, col=k)
    k+=1
fig.show()
