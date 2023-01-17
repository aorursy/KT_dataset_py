import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px



from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('../input/top-personality-dataset/2018-personality-data.csv')

df.head()
df.shape
df.columns
df = df.rename(columns={' openness' : 'openness',

                   ' agreeableness' : 'agreeableness',

                   ' emotional_stability' : 'emotional_stability',

                   ' conscientiousness' : 'conscientiousness',

                   ' extraversion' : 'extraversion',

                   ' enjoy_watching ' : 'enjoy_watching'})
ocean = ['openness','agreeableness','emotional_stability','conscientiousness','extraversion']
# pairwise scatter plot

sns.pairplot(df[ocean], kind='reg', plot_kws={'line_kws':{'color':'magenta'}, 'scatter_kws': {'alpha': 0.1}})

plt.show()
# correlation

cor_ocean = df[ocean].corr()

cor_ocean
plt.rcParams['figure.figsize']=(6,5)

sns.heatmap(cor_ocean, cmap=plt.cm.plasma)

plt.show()
scaler = StandardScaler()

df_scaled = scaler.fit_transform(df[ocean])

df_scaled
# define cluster algorithm

n_cl = 3

kmeans = KMeans(init="random", n_clusters=n_cl, n_init=10, max_iter=300, random_state=99)

# and run it

kmeans.fit(df_scaled)
# show cluster centers

kmeans.cluster_centers_
# append cluster variable

df['cluster'] = kmeans.labels_
df4pca = df[ocean]

# standardize first

df4pca_std = StandardScaler().fit_transform(df4pca)

# define 3D PCA

pc_model = PCA(n_components=3)

# apply PCA

pc = pc_model.fit_transform(df4pca_std)

# convert to data frame

df_pc = pd.DataFrame(data = pc, columns = ['pc_1', 'pc_2','pc_3'])

# add origin column

df_pc['cluster'] = df.cluster

# and look at result

df_pc.head()
# add PCA data to original data frame, so we have all data in one place

df['pc_1'] = df_pc.pc_1

df['pc_2'] = df_pc.pc_2

df['pc_3'] = df_pc.pc_3

df.head()
# interactive plot

fig = px.scatter_3d(df, x='pc_1', y='pc_2', z='pc_3',

                    color='cluster',

                    size='enjoy_watching',

                    hover_data=['userid'],

                    opacity=0.5)

fig.update_layout(title='PCA 3D')

fig.show()