import pandas as pd #Analysis 
import matplotlib.pyplot as plt #Visulization
import seaborn as sns #Visulization
import numpy as np #Analysis 
from scipy.stats import norm #Analysis 
from sklearn.preprocessing import StandardScaler #Analysis 
from scipy import stats #Analysis 
import warnings 
warnings.filterwarnings('ignore')
%matplotlib inline
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import gc
train = pd.read_csv("../input/training_set.csv")
train_meta = pd.read_csv("../input/training_set_metadata.csv")
train_meta.head()
print("unique values of object_id:",len(train_meta['object_id'].unique()),"\nshape of train_meta:",train_meta.shape[0])
#missing data
total = train_meta.isnull().sum().sort_values(ascending=False)
percent = (train_meta.isnull().sum()/train_meta.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

#histogram
#missing_data = missing_data.head(20)
percent_data = percent.head(20)
percent_data.plot(kind="bar", figsize = (10,8), fontsize = 15)
plt.xlabel("Columns", fontsize = 20)
plt.ylabel("Percent of Missing Value (%)", fontsize = 20)
#plt.title("Total Missing Value (%)", fontsize = 20)
train_meta[train_meta['distmod'].isnull()].head()
pd.isna(train_meta['distmod'].iloc[0])
train_meta['Milky.Way'] = train_meta["distmod"].apply(lambda x: 1 if pd.isnull(x) == True else 0)
train_meta.head()
f, ax = plt.subplots(figsize=(10,8))
train_meta['target'].value_counts().sort_values(ascending=False).plot.bar()
plt.show()
g = sns.FacetGrid(train_meta, hue = 'target',height=8)
g.map(plt.scatter, 'gal_l','gal_b')
g = sns.FacetGrid(train_meta, col = 'target',col_wrap=4)
g.map(plt.scatter, 'gal_l','gal_b')
g = sns.FacetGrid(train_meta, hue = 'target',height=8)
g.map(plt.scatter, 'ra','decl')
g = sns.FacetGrid(train_meta, col = 'target',col_wrap=4)
g.map(plt.scatter, 'ra','decl')
np.sin(349.046051 * np.pi / 180)
train_meta['skycoordinate_x'] = train_meta["distmod"] * train_meta["decl"].apply(lambda x: np.sin(x * np.pi/180)) * train_meta["ra"].apply(lambda x: np.cos(x * np.pi/180))
train_meta['skycoordinate_y'] = train_meta["distmod"] * train_meta["decl"].apply(lambda x: np.sin(x * np.pi/180)) * train_meta["ra"].apply(lambda x: np.sin(x * np.pi/180))
train_meta['skycoordinate_z'] = train_meta["distmod"] * train_meta["decl"].apply(lambda x: np.cos(x * np.pi/180))
#https://matplotlib.org/examples/mplot3d/scatter3d_demo.html
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10,8))
ax = Axes3D(fig)

#ax.scatter(swasp_mag1,swasp_per1,swasp_age1,edgecolor='none',c='r',marker='o',s=35,label='SWASP')
#ax.scatter(hyd_mag1,hyd_per1,hyd_age1,edgecolor='none',c='y',marker='o',s=35,label='Hyades')
#ax.scatter(pld_mag1,pld_per1,pld_age1,edgecolor='none',c='b',marker='o',s=35,label='Pleiades')

ax.scatter(train_meta[train_meta['target']==90]['skycoordinate_x'], train_meta[train_meta['target']==90]['skycoordinate_y'], train_meta[train_meta['target']==90]['skycoordinate_z'])
ax.scatter(train_meta[train_meta['target']==42]['skycoordinate_x'], train_meta[train_meta['target']==42]['skycoordinate_y'], train_meta[train_meta['target']==42]['skycoordinate_z'])
ax.scatter(train_meta[train_meta['target']==65]['skycoordinate_x'], train_meta[train_meta['target']==65]['skycoordinate_y'], train_meta[train_meta['target']==65]['skycoordinate_z'])
ax.scatter(train_meta[train_meta['target']==16]['skycoordinate_x'], train_meta[train_meta['target']==16]['skycoordinate_y'], train_meta[train_meta['target']==16]['skycoordinate_z'])
ax.scatter(train_meta[train_meta['target']==15]['skycoordinate_x'], train_meta[train_meta['target']==15]['skycoordinate_y'], train_meta[train_meta['target']==15]['skycoordinate_z'])
ax.scatter(train_meta[train_meta['target']==62]['skycoordinate_x'], train_meta[train_meta['target']==62]['skycoordinate_y'], train_meta[train_meta['target']==62]['skycoordinate_z'])
ax.scatter(train_meta[train_meta['target']==88]['skycoordinate_x'], train_meta[train_meta['target']==88]['skycoordinate_y'], train_meta[train_meta['target']==88]['skycoordinate_z'])
ax.scatter(train_meta[train_meta['target']==92]['skycoordinate_x'], train_meta[train_meta['target']==92]['skycoordinate_y'], train_meta[train_meta['target']==92]['skycoordinate_z'])
ax.scatter(train_meta[train_meta['target']==67]['skycoordinate_x'], train_meta[train_meta['target']==67]['skycoordinate_y'], train_meta[train_meta['target']==67]['skycoordinate_z'])
ax.scatter(train_meta[train_meta['target']==52]['skycoordinate_x'], train_meta[train_meta['target']==52]['skycoordinate_y'], train_meta[train_meta['target']==52]['skycoordinate_z'])
ax.scatter(train_meta[train_meta['target']==95]['skycoordinate_x'], train_meta[train_meta['target']==95]['skycoordinate_y'], train_meta[train_meta['target']==95]['skycoordinate_z'])
ax.scatter(train_meta[train_meta['target']==6]['skycoordinate_x'], train_meta[train_meta['target']==6]['skycoordinate_y'], train_meta[train_meta['target']==6]['skycoordinate_z'])
ax.scatter(train_meta[train_meta['target']==64]['skycoordinate_x'], train_meta[train_meta['target']==64]['skycoordinate_y'], train_meta[train_meta['target']==64]['skycoordinate_z'])
ax.scatter(train_meta[train_meta['target']==53]['skycoordinate_x'], train_meta[train_meta['target']==53]['skycoordinate_y'], train_meta[train_meta['target']==53]['skycoordinate_z'])

plt.title("sky coordinate")
plt.show()
train_meta['galatic_x'] = train_meta["distmod"] * train_meta["gal_b"].apply(lambda x: np.sin(x * np.pi/180)) * train_meta["gal_l"].apply(lambda x: np.cos(x * np.pi/180))
train_meta['galatic_y'] = train_meta["distmod"] * train_meta["gal_b"].apply(lambda x: np.sin(x * np.pi/180)) * train_meta["gal_l"].apply(lambda x: np.sin(x * np.pi/180))
train_meta['galatic_z'] = train_meta["distmod"] * train_meta["gal_b"].apply(lambda x: np.cos(x * np.pi/180))
#https://matplotlib.org/examples/mplot3d/scatter3d_demo.html

fig = plt.figure(figsize=(10,8))
ax = Axes3D(fig)

#ax.scatter(swasp_mag1,swasp_per1,swasp_age1,edgecolor='none',c='r',marker='o',s=35,label='SWASP')
#ax.scatter(hyd_mag1,hyd_per1,hyd_age1,edgecolor='none',c='y',marker='o',s=35,label='Hyades')
#ax.scatter(pld_mag1,pld_per1,pld_age1,edgecolor='none',c='b',marker='o',s=35,label='Pleiades')

ax.scatter(train_meta[train_meta['target']==90]['galatic_x'], train_meta[train_meta['target']==90]['galatic_y'], train_meta[train_meta['target']==90]['galatic_z'])
ax.scatter(train_meta[train_meta['target']==42]['galatic_x'], train_meta[train_meta['target']==42]['galatic_y'], train_meta[train_meta['target']==42]['galatic_z'])
ax.scatter(train_meta[train_meta['target']==65]['galatic_x'], train_meta[train_meta['target']==65]['galatic_y'], train_meta[train_meta['target']==65]['galatic_z'])
ax.scatter(train_meta[train_meta['target']==16]['galatic_x'], train_meta[train_meta['target']==16]['galatic_y'], train_meta[train_meta['target']==16]['galatic_z'])
ax.scatter(train_meta[train_meta['target']==15]['galatic_x'], train_meta[train_meta['target']==15]['galatic_y'], train_meta[train_meta['target']==15]['galatic_z'])
ax.scatter(train_meta[train_meta['target']==62]['galatic_x'], train_meta[train_meta['target']==62]['galatic_y'], train_meta[train_meta['target']==62]['galatic_z'])
ax.scatter(train_meta[train_meta['target']==88]['galatic_x'], train_meta[train_meta['target']==88]['galatic_y'], train_meta[train_meta['target']==88]['galatic_z'])
ax.scatter(train_meta[train_meta['target']==92]['galatic_x'], train_meta[train_meta['target']==92]['galatic_y'], train_meta[train_meta['target']==92]['galatic_z'])
ax.scatter(train_meta[train_meta['target']==67]['galatic_x'], train_meta[train_meta['target']==67]['galatic_y'], train_meta[train_meta['target']==67]['galatic_z'])
ax.scatter(train_meta[train_meta['target']==52]['galatic_x'], train_meta[train_meta['target']==52]['galatic_y'], train_meta[train_meta['target']==52]['galatic_z'])
ax.scatter(train_meta[train_meta['target']==95]['galatic_x'], train_meta[train_meta['target']==95]['galatic_y'], train_meta[train_meta['target']==95]['galatic_z'])
ax.scatter(train_meta[train_meta['target']==6]['galatic_x'], train_meta[train_meta['target']==6]['galatic_y'], train_meta[train_meta['target']==6]['galatic_z'])
ax.scatter(train_meta[train_meta['target']==64]['galatic_x'], train_meta[train_meta['target']==64]['galatic_y'], train_meta[train_meta['target']==64]['galatic_z'])
ax.scatter(train_meta[train_meta['target']==53]['galatic_x'], train_meta[train_meta['target']==53]['galatic_y'], train_meta[train_meta['target']==53]['galatic_z'])

plt.title("galatic coordinate")
plt.show()
train_meta['ddf'].plot()
f, ax = plt.subplots(figsize=(10,8))
train_meta['ddf'].value_counts().sort_values(ascending=False).plot.bar()
plt.show()
f, ax = plt.subplots(figsize=(10,8))
train_meta[train_meta['ddf']==1]['target'].value_counts().sort_values(ascending=False).plot.bar()
plt.title("ddf is 1 of target")
plt.show()
f, ax = plt.subplots(figsize=(10,8))
train_meta[train_meta['ddf']==0]['target'].value_counts().sort_values(ascending=False).plot.bar()
plt.title("ddf is 0 of target")
plt.show()
g = sns.FacetGrid(train_meta, hue = 'ddf',height=8)
g.map(plt.scatter, 'gal_l','gal_b')
g = sns.FacetGrid(train_meta, hue = 'ddf',height=8)
g.map(plt.scatter, 'ra','decl')
f, ax = plt.subplots(figsize=(10,8))
fig = sns.boxplot(x='target', y="hostgal_specz", data=train_meta)
f, ax = plt.subplots(figsize=(10,8))
fig = sns.boxplot(x='target', y="hostgal_photoz", data=train_meta)
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(14,12),sharey=True)
sns.boxplot(x='target', y="hostgal_specz", data=train_meta,ax=axis1)
axis1.set_title('Spectroscopic Redshift')
sns.boxplot(x='target', y="hostgal_photoz", data=train_meta,ax=axis2)
axis2.set_title('Photometric Redshift')
f, ax = plt.subplots(figsize=(10,8))
fig = sns.boxplot(x='target', y="distmod", data=train_meta)
train_meta[train_meta['distmod'].isnull()]['target'].unique()
g = sns.FacetGrid(train_meta, hue = 'Milky.Way',height=8)
g.map(plt.scatter, 'gal_l','gal_b')
g = sns.FacetGrid(train_meta, hue = 'Milky.Way',height=8)
g.map(plt.scatter, 'ra','decl')
g = sns.FacetGrid(train_meta, col = 'Milky.Way',col_wrap=2)
g.map(plt.scatter, 'gal_l','gal_b')
g = sns.FacetGrid(train_meta, col = 'Milky.Way',col_wrap=2)
g.map(plt.scatter, 'ra','decl')
print("Milky.Way is 0 of Target:",train_meta[train_meta['Milky.Way']==0]['target'].unique(),"\nMilky.Way is 1 of Target:",train_meta[train_meta['Milky.Way']==1]['target'].unique())
train.head()
train.describe()
train['object_id'] = train['object_id'].astype(np.object)
train_meta['object_id'] = train_meta['object_id'].astype(np.object)
train['object_id'].head()
train_all = train_meta.merge(train, how='left', on='object_id')
train_all.head()
del train, train_meta
gc.collect()
train_all['mjd'].head()
train_all['date'] = pd.to_datetime((train_all['mjd']-40587)*86400,unit='s')
train_all.head()
train_all['target'].unique()
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff
import random

#https://www.kaggle.com/ashishpatel26/bird-eye-view-of-two-sigma-xgb
def generate_color():
    color = '#{:02x}{:02x}{:02x}'.format(*map(lambda x: random.randint(0, 255), range(3)))
    return color
for i in [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]:
    
    data = []
    for j in [0,1,2,3,4,5]:
        j = go.Scatter(
                x = train_all[(train_all['target']==i)]['date'].dt.strftime(date_format='%Y-%m-%d').values,
                y = train_all[(train_all['target']==i) & (train_all['passband']==j)]['flux'].values,
                line = dict(color = generate_color()),opacity = 0.8
            )

        data.append(j)
                    
    layout = dict(title = "Target {}".format(i),
                          xaxis = dict(title = 'date'),
                          yaxis = dict(title = 'flux'),
                          )
    py.iplot(dict(data=data, layout=layout), filename='basic-line')
fig, (axes) = plt.subplots(2,3,figsize=(18,12),sharey=True)
sns.boxplot(x='target', y="flux_err", data=train_all[train_all['passband']==0],ax=axes[0,0])
axes[0,0].set_title('0')
sns.boxplot(x='target', y="flux_err", data=train_all[train_all['passband']==1],ax=axes[0,1])
axes[0,1].set_title('1')
sns.boxplot(x='target', y="flux_err", data=train_all[train_all['passband']==2],ax=axes[0,2])
axes[0,2].set_title('2')
sns.boxplot(x='target', y="flux_err", data=train_all[train_all['passband']==3],ax=axes[1,0])
axes[1,0].set_title('3')
sns.boxplot(x='target', y="flux_err", data=train_all[train_all['passband']==4],ax=axes[1,1])
axes[1,1].set_title('4')
sns.boxplot(x='target', y="flux_err", data=train_all[train_all['passband']==5],ax=axes[1,2])
axes[1,2].set_title('5')
f, ax = plt.subplots(figsize=(10,8))
train_all['detected'].value_counts().sort_values(ascending=False).plot.bar()
plt.show()
fig, (axes) = plt.subplots(2,2,figsize=(18,12))
ax1 = plt.subplot(2,2,1)
train_all[(train_all['detected']==0) & (train_all['ddf']==0)]['target'].value_counts().sort_values(ascending=False).plot.bar()
plt.title('detected : 0 & ddf : 0')
ax1 = plt.subplot(2,2,2)
train_all[(train_all['detected']==0) & (train_all['ddf']==1)]['target'].value_counts().sort_values(ascending=False).plot.bar()
plt.title('detected : 0 & ddf : 1')
ax1 = plt.subplot(2,2,3)
train_all[(train_all['detected']==1) & (train_all['ddf']==0)]['target'].value_counts().sort_values(ascending=False).plot.bar()
plt.title('detected : 1 & ddf : 0')
ax1 = plt.subplot(2,2,4)
train_all[(train_all['detected']==1) & (train_all['ddf']==1)]['target'].value_counts().sort_values(ascending=False).plot.bar()
plt.title('detected : 1 & ddf : 1')
fig, (axes) = plt.subplots(2,2,figsize=(18,12))
ax1 = plt.subplot(2,2,1)
train_all[(train_all['detected']==0) & (train_all['Milky.Way']==0)]['target'].value_counts().sort_values(ascending=False).plot.bar()
plt.title('detected : 0 & Milky.Way : 0')
ax1 = plt.subplot(2,2,2)
train_all[(train_all['detected']==0) & (train_all['Milky.Way']==1)]['target'].value_counts().sort_values(ascending=False).plot.bar()
plt.title('detected : 0 & Milky.Way : 1')
ax1 = plt.subplot(2,2,3)
train_all[(train_all['detected']==1) & (train_all['Milky.Way']==0)]['target'].value_counts().sort_values(ascending=False).plot.bar()
plt.title('detected : 1 & Milky.Way : 0')
ax1 = plt.subplot(2,2,4)
train_all[(train_all['detected']==1) & (train_all['Milky.Way']==1)]['target'].value_counts().sort_values(ascending=False).plot.bar()
plt.title('detected : 1 & Milky.Way : 1')
fig, (axes) = plt.subplots(2,2,figsize=(18,12))
ax1 = plt.subplot(2,2,1)
train_all[(train_all['ddf']==0) & (train_all['Milky.Way']==0)]['target'].value_counts().sort_values(ascending=False).plot.bar()
plt.title('ddf : 0 & Milky.Way : 0')
ax1 = plt.subplot(2,2,2)
train_all[(train_all['ddf']==0) & (train_all['Milky.Way']==1)]['target'].value_counts().sort_values(ascending=False).plot.bar()
plt.title('ddf : 0 & Milky.Way : 1')
ax1 = plt.subplot(2,2,3)
train_all[(train_all['ddf']==1) & (train_all['Milky.Way']==0)]['target'].value_counts().sort_values(ascending=False).plot.bar()
plt.title('ddf : 1 & Milky.Way : 0')
ax1 = plt.subplot(2,2,4)
train_all[(train_all['ddf']==1) & (train_all['Milky.Way']==1)]['target'].value_counts().sort_values(ascending=False).plot.bar()
plt.title('ddf : 1 & Milky.Way : 1')
