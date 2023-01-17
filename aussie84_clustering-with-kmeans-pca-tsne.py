# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import plotly_express as px

import plotly.graph_objs as go

import plotly.plotly as py

# from plotly.offline import init_notebook_mode, iplot

# init_notebook_mode(connected=True)





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/data.csv')
df.columns
skill_cols = ['Crossing',

       'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',

       'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',

       'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',

       'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',

       'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',

       'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',

       'GKKicking', 'GKPositioning', 'GKReflexes']

dfskills = df[skill_cols]

len(dfskills.columns)
import pandas_profiling

# pandas_profiling.ProfileReport(dfskills)
dfskills.dropna(inplace=True)
corr = dfskills.corr()

trace = go.Heatmap(z=corr,x=corr.index,y=corr.columns)

data = [trace]

layout = dict(title="Correlation Plot of Player Skills")

fig = dict(data=data, layout=layout)

#iplot(fig)
from sklearn.decomposition import PCA

pca = PCA().fit(dfskills)
pcaratio = pca.explained_variance_ratio_

trace = go.Scatter(x=np.arange(len(pcaratio)),y=np.cumsum(pcaratio))

data = [trace]

layout = dict(title="Player Skills Dataset - PCA Explained Variance || 89% achieved at 5 components")

fig = dict(data=data, layout=layout)

#iplot(fig)
pca = PCA(n_components=5)

skillsPCA = pca.fit_transform(dfskills)
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=6)

skillsPCA_labels = kmeans.fit_predict(skillsPCA)
dfskillsPCA = pd.DataFrame(skillsPCA)

dfskillsPCA['cluster'] = skillsPCA_labels
from sklearn.manifold import TSNE

X = dfskillsPCA.iloc[:,:-1]

Xtsne = TSNE(n_components=2).fit_transform(X)

dftsne = pd.DataFrame(Xtsne)

dftsne['cluster'] = skillsPCA_labels

dftsne.columns = ['x1','x2','cluster']
pca2 = PCA(n_components=2)

skillsPCA2 = pca2.fit_transform(dfskills)

dfskillsPCA2 = pd.DataFrame(skillsPCA2)

dfskillsPCA2['cluster'] = skillsPCA_labels

dfskillsPCA2.columns = ['x1','x2','cluster']

fig, ax = plt.subplots(1, 2, figsize=(12,6))

sns.scatterplot(data=dftsne,x='x1',y='x2',hue='cluster',legend="full",alpha=0.5,ax=ax[0])

ax[0].set_title('Visualized on TSNE 2D')

sns.scatterplot(data=dfskillsPCA2,x='x1',y='x2',hue='cluster',legend="full",alpha=0.5,ax=ax[1])

ax[1].set_title('Visualized on PCA 2D')

fig.suptitle('Comparing clustering result when visualized using TSNE2D vs. PCA2D')

display(fig)
kmeans = KMeans(n_clusters=6)

clustering_ori = kmeans.fit_predict(dfskills)
dftsne2D = dftsne

dftsne2D['cluster'] = clustering_ori
X = dfskills

Xtsne = TSNE(n_components=2).fit_transform(X)

dftsneFull = pd.DataFrame(Xtsne)
dftsneFull['cluster'] = clustering_ori

dftsneFull.columns = ['x1','x2','cluster']
fig, ax = plt.subplots(1, 2, figsize=(12,6))

sns.scatterplot(data=dftsne2D,x='x1',y='x2',hue='cluster',legend="full",alpha=0.5,ax=ax[0])

ax[0].set_title('Visualized on TSNE 5D>2D')

sns.scatterplot(data=dftsneFull,x='x1',y='x2',hue='cluster',legend="full",alpha=0.7,ax=ax[1])

ax[1].set_title('Visualized on TSNE 34D>2D')

fig.suptitle('Comparing clustering result when visualized using TSNE 5D>2D vs. TSNE 34D>2D')

display(fig)
dfskills['cluster'] = clustering_ori
# Some functions to plot just the variables that has significant deviation from global mean

def outside_limit(df, label_col, label, sensitivity):

  feature_list = dfskills.columns[:-1]

  

  plot_list = []

  mean_overall_list = []

  mean_cluster_list = []

  

  for i,varname in enumerate(feature_list):

    

    #     get overall mean for a variable, set lower and upper limit

    mean_overall = df[varname].mean()

    lower_limit = mean_overall - (mean_overall*sensitivity)

    upper_limit = mean_overall + (mean_overall*sensitivity)



    #     get cluster mean for a variable

    cluster_filter = df[label_col]==label

    pd_cluster = df[cluster_filter]

    mean_cluster = pd_cluster[varname].mean()

    

    #     create filter to display graph with 0.5 deviation from the mean

    if mean_cluster <= lower_limit or mean_cluster >= upper_limit:

      plot_list.append(varname)

      mean_overall_std = mean_overall/mean_overall

      mean_cluster_std = mean_cluster/mean_overall

      mean_overall_list.append(mean_overall_std)

      mean_cluster_list.append(mean_cluster_std)

   

  mean_df = pd.DataFrame({'feature_list':plot_list,

                         'mean_overall_list':mean_overall_list,

                         'mean_cluster_list':mean_cluster_list})

  mean_df = mean_df.sort_values(by=['mean_cluster_list'], ascending=False)

  

  return mean_df



def plot_barchart_all_unique_features(df, label_col, label, ax, sensitivity):

  

  mean_df = outside_limit(df, label_col, label, sensitivity)

  mean_df_to_plot = mean_df.drop(['mean_overall_list'], axis=1)

  

  if len(mean_df.index) != 0:

    sns.barplot(y='feature_list', x='mean_cluster_list', data=mean_df_to_plot, palette=sns.cubehelix_palette(20, start=.5, rot=-.75, reverse=True), \

                alpha=0.75, dodge=True, ax=ax)



    for i,p in enumerate(ax.patches):

      ax.annotate("{:.02f}".format((p.get_width())), 

                  (1, p.get_y() + p.get_height() / 2.), xycoords=('axes fraction', 'data'),

                  ha='right', va='top', fontsize=10, color='black', rotation=0, 

                  xytext=(0, 0),

                  textcoords='offset pixels')

  

  ax.set_title('Unique Characteristics of Cluster ' + str(label))

  ax.set_xlabel('Standardized Mean')

  ax.axvline(x=1, color='k')



def plot_features_all_cluster(df, label_col, n_clusters, sensitivity):

  n_plot = n_clusters

  fig, ax = plt.subplots(n_plot, 1, figsize=(12, n_plot*6), sharex='col')

  ax= ax.ravel()

  

  label = np.arange(n_clusters)

  for i in label:

    plot_barchart_all_unique_features(df, label_col, label=i, ax=ax[i], sensitivity=sensitivity)

    ax[i].xaxis.set_tick_params(labelbottom=True)

    

  plt.tight_layout()

  display(fig)
plot_features_all_cluster(df=dfskills, label_col='cluster', n_clusters=6, sensitivity=0.2)