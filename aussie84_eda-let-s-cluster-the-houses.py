import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import pandas as pd

from datetime import datetime

from scipy.stats import skew

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax

print(os.listdir("../input"))

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

print("Train set size:", train.shape)

print("Test set size:", test.shape)

print('START data processing', datetime.now(), )

train_ID = train['Id']

test_ID = test['Id']

# Now drop the  'Id' colum since it's unnecessary for  the prediction process.

train.drop(['Id'], axis=1, inplace=True)

test.drop(['Id'], axis=1, inplace=True)

print(train.shape)
# Deleting outliers

train = train[train.GrLivArea < 4500]

train.reset_index(drop=True, inplace=True)



# We use the numpy fuction log1p which  applies log(1+x) to all elements of the column

train["SalePrice"] = np.log1p(train["SalePrice"])

y = train.SalePrice.reset_index(drop=True)

train_features = train.drop(['SalePrice'], axis=1)

test_features = test



features = pd.concat([train_features, test_features]).reset_index(drop=True)

print(features.shape)

# Some of the non-numeric predictors are stored as numbers; we convert them into strings 



features['MSSubClass'] = features['MSSubClass'].apply(str)

features['YrSold'] = features['YrSold'].astype(str)

features['MoSold'] = features['MoSold'].astype(str)



features['Functional'] = features['Functional'].fillna('Typ')

features['Electrical'] = features['Electrical'].fillna("SBrkr")

features['KitchenQual'] = features['KitchenQual'].fillna("TA")

features['Exterior1st'] = features['Exterior1st'].fillna(features['Exterior1st'].mode()[0])

features['Exterior2nd'] = features['Exterior2nd'].fillna(features['Exterior2nd'].mode()[0])

features['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])



features["PoolQC"] = features["PoolQC"].fillna("None")



for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    features[col] = features[col].fillna(0)

for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:

    features[col] = features[col].fillna('None')

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    features[col] = features[col].fillna('None')



features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))



objects = []

for i in features.columns:

    if features[i].dtype == object:

        objects.append(i)



features.update(features[objects].fillna('None'))



features['LotFrontage'] = features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))



# Filling in the rest of the NA's



numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numerics = []

for i in features.columns:

    if features[i].dtype in numeric_dtypes:

        numerics.append(i)

features.update(features[numerics].fillna(0))



numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numerics2 = []

for i in features.columns:

    if features[i].dtype in numeric_dtypes:

        numerics2.append(i)



skew_features = features[numerics2].apply(lambda x: skew(x)).sort_values(ascending=False)



high_skew = skew_features[skew_features > 0.5]

skew_index = high_skew.index



for i in skew_index:

    features[i] = boxcox1p(features[i], boxcox_normmax(features[i] + 1))



features = features.drop(['Utilities', 'Street', 'PoolQC',], axis=1)



features['YrBltAndRemod']=features['YearBuilt']+features['YearRemodAdd']

features['TotalSF']=features['TotalBsmtSF'] + features['1stFlrSF'] + features['2ndFlrSF']



features['Total_sqr_footage'] = (features['BsmtFinSF1'] + features['BsmtFinSF2'] +

                                 features['1stFlrSF'] + features['2ndFlrSF'])



features['Total_Bathrooms'] = (features['FullBath'] + (0.5 * features['HalfBath']) +

                               features['BsmtFullBath'] + (0.5 * features['BsmtHalfBath']))



features['Total_porch_sf'] = (features['OpenPorchSF'] + features['3SsnPorch'] +

                              features['EnclosedPorch'] + features['ScreenPorch'] +

                              features['WoodDeckSF'])



# simplified features

features['haspool'] = features['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

features['has2ndfloor'] = features['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

features['hasgarage'] = features['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

features['hasbsmt'] = features['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

features['hasfireplace'] = features['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)



print(features.shape)

final_features = pd.get_dummies(features).reset_index(drop=True)

print(final_features.shape)
X = final_features.iloc[:len(y), :]

X_sub = final_features.iloc[len(X):, :]

print('X', X.shape, 'y', y.shape, 'X_sub', X_sub.shape)

outliers = [30, 88, 462, 631, 1322]

X = X.drop(X.index[outliers])

y = y.drop(y.index[outliers])

overfit = []

for i in X.columns:

    counts = X[i].value_counts()

    zeros = counts.iloc[0]

    if zeros / len(X) * 100 > 99.94:

        overfit.append(i)



overfit = list(overfit)

overfit.append('MSZoning_C (all)')



X = X.drop(overfit, axis=1).copy()

X_sub = X_sub.drop(overfit, axis=1).copy()



print('X', X.shape, 'y', y.shape, 'X_sub', X_sub.shape)

# ################## ML ########################################

print('START ML', datetime.now(), )
import plotly.graph_objs as go

import plotly.plotly as py

import plotly.offline as pyo

from plotly import tools

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly_express as px

init_notebook_mode(connected=True)

from matplotlib import cm

from matplotlib import pyplot as plt

import seaborn as sns



from sklearn.decomposition import PCA

from sklearn.cluster import KMeans
X.head()
pca = PCA(n_components=50).fit(X)

#Plotting the Cumulative Summation of the Explained Variance

expvar=np.cumsum(pca.explained_variance_ratio_)

data = [go.Scatter(y=expvar)]

layout = {'title': 'Review PCA Explained Variance to determine number of components'}

iplot({'data':data,'layout':layout})
pca = PCA(n_components=4)

XPCA = pca.fit_transform(X)
Nc = range(1,20)

kmeans = [KMeans(i) for i in Nc]

score = [kmeans[i].fit(XPCA).score(XPCA) for i in range(len(kmeans))]
data = [go.Scatter(y=score,x=list(Nc))]

layout = {'title':'Review Elbow Curve to determine number of clusters for KMeans'}

iplot({'data':data,'layout':layout})
n_clusters=5

kmeans = KMeans(n_clusters=n_clusters, random_state=42)

Xkmeans = kmeans.fit_predict(XPCA)
from sklearn.manifold import TSNE

XTSNE = TSNE(n_components=2).fit_transform(XPCA)
y.shape
XTSNEdf = pd.concat([pd.DataFrame(XTSNE),pd.DataFrame(Xkmeans),pd.DataFrame(y),pd.DataFrame(np.expm1(y))],axis=1)

XTSNEdf.columns = ['x1','x2','cluster','logprice','price']

px.scatter(XTSNEdf,x='x1',y='x2',color='cluster',color_continuous_scale=px.colors.qualitative.Plotly,title="TSNE visualization of House Clusters",width=800,height=500)
XTSNEdf.head()
px.scatter(XTSNEdf,x='x1',y='x2',color='price',hover_data=['price'], color_continuous_scale=px.colors.colorbrewer.Greens,title="TSNE visualization to check relationship with price",width=1000,height=600)
px.density_contour(XTSNEdf, x="x1", y="x2", title="Contour plot to see distribution of data")
px.scatter_3d(XTSNEdf, x="x1", y="x2", z="price", color='cluster',color_continuous_scale=px.colors.qualitative.Plotly, title="3D plotting of price against x1 and x2")
px.violin(XTSNEdf, x="cluster", y="price",box=True, points='all', title="Violin plot to compare price distribution between clusters")
def outside_limit(df, label_col, label, feature_list):

  

  plot_list = []

  mean_overall_list = []

  mean_cluster_list = []

  

  for i,varname in enumerate(feature_list):

    

    #     get overall mean for a variable, set lower and upper limit

    mean_overall = df[varname].mean()

    lower_limit = mean_overall - (mean_overall*0.7)

    upper_limit = mean_overall + (mean_overall*0.7)



    #     get cluster mean for a variable

    cluster_filter = df[label_col]==label

    pd_cluster = df[cluster_filter]

    mean_cluster = pd_cluster[varname].mean()

    

    #     create filter to display graph with 0.5 deviation from the mean

    if (mean_cluster <= lower_limit or mean_cluster >= upper_limit) and mean_cluster != 0:

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



def plot_barchart_all_unique_features(df, label_col, feature_list, label, ax):

  mean_df = outside_limit(df, label_col, label, feature_list)

  mean_df_to_plot = mean_df.drop(['mean_overall_list'], axis=1)

  

  plot_list = list(mean_df_to_plot.feature_list)

  char = df.groupby(label_col)[feature_list].mean().reset_index(level=0,drop=True).reset_index()

#   char = char.loc[char[label_col] == label, plot_list]

  

  if len(mean_df.index) != 0:

    sns.barplot(y='feature_list', x='mean_cluster_list', data=mean_df_to_plot, color="maroon", \

                alpha=0.75, dodge=True, ax=ax)



    for i,p in enumerate(ax.patches):

      ax.annotate("{:.02f}".format((p.get_width())), 

                  (0.925, p.get_y() + p.get_height() / 2.), xycoords=('axes fraction', 'data'),

                  ha='right', va='top', fontsize=10, color='black', rotation=0, 

                  xytext=(0, 0),

                  textcoords='offset pixels')

      

#       ax.annotate("({0}%)".format(round(char.iloc[:,i],4)*100), 

#                   (1.02, p.get_y() + p.get_height() / 2.), xycoords=('axes fraction', 'data'),

#                   ha='right', va='top', fontsize=10, color='black', rotation=0, 

#                   xytext=(0, 0),

#                   textcoords='offset pixels')

      

  

  ax.set_title('Unique Characteristics of Cluster ' + str(label))

  ax.set_xlabel('Standardized Mean')

  ax.axvline(x=1, color='k')
XTSNEall = pd.concat([pd.DataFrame(Xkmeans),pd.DataFrame(X),pd.DataFrame(y),pd.DataFrame(np.expm1(y))],axis=1)

XTSNEall.columns = ['cluster'] + list(X.columns) + ['logprice','price']
sns.set_context("paper", font_scale=2) 

numclusters = 5

label = list(range(numclusters))

fig, ax = plt.subplots(numclusters,1,figsize=(8,20*numclusters))

for i in range(numclusters):

    plot_barchart_all_unique_features(XTSNEall, 'cluster', list(XTSNEall.columns)[1:], i, ax[i])
def visualize_dimension_reduction_per_cluster(type, df_components, cluster_no, labels, ax, cluster_desc):

  

  color_list = sns.color_palette("Paired")

#   plt.clf()

#   plt.cla()

#   plt.close()



#   f, ax = plt.subplots(1,1, figsize = (20,20))

  



  for l in set(labels):

    

    if l == cluster_no:

      sns.scatterplot(df_components.loc[df_components['cluster'] == l, 'x1'], df_components.loc[df_components['cluster'] == l, 'x2'], color = color_list[l], ax = ax, label=str(cluster_no))

      ax.set_title("Cluster " + str(l) + " : " + cluster_desc[l])

    else:

      sns.scatterplot(df_components.loc[df_components['cluster'] == l, 'x1'], df_components.loc[df_components['cluster'] == l, 'x2'], color = "grey", ax = ax)

#   L=plt.legend()

#   if time_of_day == "Morning":

#     L.get_texts()[0].set_text('Current date {}'.format(after_date))

#     L.get_texts()[1].set_text('Previous date {}'.format(before_date))

#   else:

#     L.get_texts()[0].set_text('Current date {}'.format(after_time))

#     L.get_texts()[1].set_text('Current date {}'.format(before_time))

#   display(f)



def plot_features_all_cluster(df, label_col, feature_list, label, cluster_desc):

  plt.clf()

  plt.cla()

  plt.close()

  

  fig, ax = plt.subplots(len(label), 2, figsize=(16,15*numclusters), sharex='col')

#   ax= ax.ravel()

  

#   label = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

  for i in label:

#   for l in set(label):

#     j = l % 2

#     i = (l - j)/2

    plot_barchart_all_unique_features(df, label_col, feature_list, label=i, ax=ax[i,1])

    ax[i,1].xaxis.set_tick_params(labelbottom=True)

    

    

    visualize_dimension_reduction_per_cluster("TSNE", df, i, label, ax[i,0], cluster_desc)

  

    

    

  plt.tight_layout()

  plt.subplots_adjust(hspace = 0.4)

  display(fig)
feature_columns = list(X.columns)

XTSNEall2 = pd.concat([pd.DataFrame(XTSNE),XTSNEall],axis=1)

XTSNEall2.columns = ['x1','x2'] + list(XTSNEall.columns)

sns.set_context("paper", font_scale=1.5) 

cluster_desc = {0: "Generic", 

                1: "So Warm", 

                2: "Some Road Condition?", 

                3: "RRNn", 

                4: "Exterior"}

plot_features_all_cluster(XTSNEall2, 'cluster', feature_columns, label, cluster_desc)
n_clusters=10

kmeans = KMeans(n_clusters=n_clusters, random_state=42)

Xkmeans = kmeans.fit_predict(XPCA)

XTSNE = TSNE(n_components=2).fit_transform(XPCA)

XTSNEdf = pd.concat([pd.DataFrame(XTSNE),pd.DataFrame(Xkmeans),pd.DataFrame(y),pd.DataFrame(np.expm1(y))],axis=1)

XTSNEdf.columns = ['x1','x2','cluster','logprice','price']

px.scatter(XTSNEdf,x='x1',y='x2',color='cluster',color_continuous_scale=px.colors.qualitative.Plotly,title="TSNE visualization of House Clusters",width=800,height=500)
XTSNEall = pd.concat([pd.DataFrame(Xkmeans),pd.DataFrame(X),pd.DataFrame(y),pd.DataFrame(np.expm1(y))],axis=1)

XTSNEall.columns = ['cluster'] + list(X.columns) + ['logprice','price']



numclusters = 10

label = list(range(numclusters))

feature_columns = list(X.columns)

XTSNEall2 = pd.concat([pd.DataFrame(XTSNE),XTSNEall],axis=1)

XTSNEall2.columns = ['x1','x2'] + list(XTSNEall.columns)

sns.set_context("paper", font_scale=1.5) 

cluster_desc = {}

for i in range(numclusters):

    cluster_desc[i] = str(i)

plot_features_all_cluster(XTSNEall2, 'cluster', feature_columns, label, cluster_desc)