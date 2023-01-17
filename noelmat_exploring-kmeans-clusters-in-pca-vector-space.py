import numpy as np

import pandas as pd

from pathlib import Path

Path.ls = lambda x: list(x.iterdir())

from sklearn.preprocessing import MinMaxScaler

from sklearn.decomposition import PCA

from matplotlib import pyplot as plt

import seaborn as sns

import plotly.express as px

import math

from scipy import stats

from sklearn.cluster import KMeans
path = Path('/kaggle/input/ccdata/')

path.ls()
df = pd.read_csv(path/'CC GENERAL.csv')

df.head()
df.describe()
df.info(), df.isna().sum(), df.isna().sum()/len(df)
na_cols = df.columns[df.isna().sum() > 0].tolist()

df.loc[:,na_cols] = df.loc[:,na_cols].fillna(df[na_cols].median())
df.isna().sum().sum()
# I am too lazy to write down all the columns for cont cols ;) 

cat_cols = ['TENURE']

cont_cols = df.columns.tolist()

cont_cols.remove(cat_cols[0])

cont_cols.remove('CUST_ID')
def plot_univariable_plots(df, cat_cols, cont_cols):

    total_cols = len(cat_cols)+len(cont_cols)

    fig, axes = plt.subplots(math.ceil(total_cols/3),3, figsize=(20,20),constrained_layout=True)

    axes = axes.flatten()

    fig.suptitle(f'Univariate plots'.title(),fontsize=18)

    

    for i, (col, ax) in enumerate(zip(cont_cols, axes)):

        sns.distplot(df[col], ax=ax)

        ax.set_title(f'Histogram of {col}')

    

    for col in cat_cols:

        sns.countplot(df[col],ax=axes[i+1])

        ax.set_title(f'Histogram of {col}')

        

    plt.show()
plot_univariable_plots(df, cat_cols, cont_cols)
transformed_df = df.copy()

transformed_df.loc[:,cont_cols] = transformed_df[cont_cols].apply(lambda x: stats.boxcox(x+1)[0], axis=0)

plot_univariable_plots(transformed_df, cat_cols, cont_cols)
scaler = MinMaxScaler()

scaler.fit(transformed_df[cont_cols+cat_cols])

scaled = scaler.transform(transformed_df[cont_cols+cat_cols])

scaled_df = pd.DataFrame(scaled, columns=cont_cols+cat_cols)
N_COMPONENTS = 15

pca = PCA(n_components=N_COMPONENTS)

pca.fit(scaled_df)

pca.explained_variance_ratio_[:4].sum()
pca_data = pca.transform(scaled_df)

pca_df = pd.DataFrame(pca_data).iloc[:,:4]

pca_df.columns = list(map(lambda x: f'pca_{x+1}', pca_df.columns))

# pca_df['TENURE'] = df.TENURE
fig = px.scatter_3d(pca_df,x='pca_1',y='pca_2',z='pca_3',opacity=0.3,color='pca_4')

fig.show()
cost = []

ks = []

for i in range(3,30):

    kmeans = KMeans(n_clusters=i)

    kmeans.fit(pca_df)

    cost.append(kmeans.inertia_)

    ks.append(i)

sns.lineplot(x=np.array(ks), y=np.array(cost))

plt.xticks(ks)

plt.show()
kmeans = KMeans(n_clusters=5)

kmeans.fit(pca_df)

out = kmeans.predict(pca_df)
fig = px.scatter_3d(pca_df,x='pca_1',y='pca_2',z='pca_3',color=out,opacity=0.5,

                    title='KMeans cluster with k=5')

fig.show()
def display_component(v, features_list, component_num,ax):

    

    row_idx = component_num

    

    v_1_row = v.iloc[:,row_idx]

    v_1 = np.squeeze(v_1_row.values)

    

    comps = pd.DataFrame(list(zip(v_1, features_list)),

                         columns=['weights', 'features'])

    

    comps['abs_weights']=comps['weights'].apply(lambda x: np.abs(x))

    sorted_weight_data = comps.sort_values('abs_weights',ascending=False).head()

    

    sns.barplot(data=sorted_weight_data,

                   x="weights",

                   y="features",

                   palette="Blues_d",ax=ax)

    ax.set_title("PCA Component Makeup, Component #" + str(component_num), fontsize=20)

features_list = np.array(cont_cols+cat_cols)

v = pd.DataFrame(pca.components_)
fig, axes = plt.subplots(2,2,figsize=(20,8),constrained_layout=True)

axes=axes.flatten()

for i,ax in enumerate(axes):

    display_component(v, features_list, i,ax=ax)

plt.show()
cluster_centers = kmeans.cluster_centers_

behaviours = cluster_centers.dot(v[:4])
fig, axes = plt.subplots(3,2,figsize=(15,12),constrained_layout=True)

axes=axes.flatten()

threshold = 0.2

for i,behaviour in enumerate(behaviours):

    thresh_mask = np.nonzero(np.abs(behaviour)>threshold)[0].tolist()

    sns.barplot(behaviour[thresh_mask], y=features_list[thresh_mask],ax=axes[i])

    axes[i].set_title(f'Cluster {i+1} features')

plt.show()
