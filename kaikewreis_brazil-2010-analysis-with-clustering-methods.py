import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
# Import data with pandas

df = pd.read_csv('../input/brazilian-cities/BRAZIL_CITIES.csv',delimiter=';')

# Select our columns based in 2010 measures

desired_cols = list(df.columns[0:16]) + list(df.columns[19:25])

df = df[desired_cols]
# Preview

df.head(3)
# Plot a Brazil map based in LAT/LON

## remove zero values

mask1= df["LONG"] != 0

mask2 = df["LAT"] !=0 

mask3 = df['CAPITAL'] ==1

 

## use the scatter function

plt.figure(figsize=(10,10))

plt.title("Cities Latitude and Longitude")

plt.xlabel("Longitude")

plt.ylabel("Latitude")

plt.scatter(df[mask1&mask2&mask3]["LONG"], df[mask1&mask2&mask3]["LAT"], s=20, alpha=1, label='Capital city')

plt.scatter(df[mask1&mask2&~mask3]["LONG"], df[mask1&mask2&~mask3]["LAT"], s=1, alpha=1, label='Other')

plt.legend()

plt.show()
# Loop into all samples to create the categories

for i in range(0, len(df)):

    # Category 1

    if df.loc[i,'IDHM'] >= 0.0 and df.loc[i,'IDHM'] < 0.500:

        df.loc[i,'IDHM_CAT'] = 'C1'

    # Category 2

    elif df.loc[i,'IDHM'] >= 0.500 and df.loc[i,'IDHM'] < 0.600:

        df.loc[i,'IDHM_CAT'] = 'C2'

    # Category 3

    elif df.loc[i,'IDHM'] >= 0.600 and df.loc[i,'IDHM'] < 0.700:

        df.loc[i,'IDHM_CAT'] = 'C3'

    # Category 4

    elif df.loc[i,'IDHM'] >= 0.700 and df.loc[i,'IDHM'] < 0.800:

        df.loc[i,'IDHM_CAT'] = 'C4'

    # Category 5

    elif df.loc[i,'IDHM'] >= 0.800 and df.loc[i,'IDHM'] <= 1:

        df.loc[i,'IDHM_CAT'] = 'C5'
# Barplot

sns.countplot(x="IDHM_CAT", data=df, order=['C1','C2','C3','C4','C5']);
# Loop into all samples to create the categories more balanced

for i in range(0, len(df)):

    # Category 1 and 2

    if df.loc[i,'IDHM'] >= 0.0 and df.loc[i,'IDHM'] < 0.600:

        df.loc[i,'IDHM_CAT2'] = 'C1-C2'

    # Category 3

    elif df.loc[i,'IDHM'] >= 0.600 and df.loc[i,'IDHM'] < 0.700:

        df.loc[i,'IDHM_CAT2'] = 'C3'

    # Category 4

    elif df.loc[i,'IDHM'] >= 0.700 and df.loc[i,'IDHM'] <= 1:

        df.loc[i,'IDHM_CAT2'] = 'C4-C5'
# Barplot

sns.countplot(x="IDHM_CAT2", data=df, order=['C1-C2','C3','C4-C5']);
# See how many NaN values do I have

df.isnull().sum()
# How many rows do I have now?

sb = len(df)

sb
# Drop any row that contains a single NaN value

df.dropna(axis=0, inplace=True)
# Reset index

df.index = range(0,len(df))
# How many rows do I have after cleaning my dataset?

sa = len(df)

sa
# What I have lost?

sb - sa
# Separate y categorical (targets)

y = df[['IDHM_CAT2']]

yReal = df[['IDHM_CAT']]

# Separate y numeric

yNum = df[['IDHM']]

# Separate x (predictors)

x = df[['IBGE_RES_POP', 'IBGE_RES_POP_BRAS','IBGE_RES_POP_ESTR', 'IBGE_DU', 'IBGE_DU_URBAN', 'IBGE_DU_RURAL', 'IBGE_POP', 

        'IBGE_1', 'IBGE_1-4', 'IBGE_5-9', 'IBGE_10-14', 'IBGE_15-59', 'IBGE_60+']]

# Separate a (analysis)

a = df[['CITY', 'STATE', 'CAPITAL', 'LONG', 'LAT']]
# Import sklearn preprocessing library

from sklearn.preprocessing import StandardScaler
# Create MinMaxScaler object

normData = StandardScaler()
# Scale our x set by applying a fit and transform

nd_x = normData.fit_transform(x)
# Transform the results into a dataframe as was the original

x = pd.DataFrame(nd_x, index=x.index, columns=x.columns)
# Spearman correlation

mc = x.corr(method='spearman')
# Generate a mask for the upper triangle

triangle_mask = np.zeros_like(mc, dtype=np.bool)

triangle_mask[np.triu_indices_from(triangle_mask)] = True



# Plot

plt.figure(figsize = (15,15))

sns.heatmap(data = mc, linewidths=.1, linecolor='black', vmin = -1, vmax = 1, mask = triangle_mask, annot = True,

            cbar_kws={"ticks":[-1,-0.8,-0.6,0,0.6,0.8,1]});
# Import the libraries to FS and score function (metric)

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import mutual_info_classif
# FS process for only two predictors variables

FS2 = SelectKBest(score_func=mutual_info_classif, k=2);
# Fit and Transform

x_fs2 = FS2.fit_transform(x, y.values.ravel());
# The chosen 2 variables

fs2_cols = x.columns[FS2.get_support()]

fs2_cols
# FS process for three predictors variables

FS3 = SelectKBest(score_func=mutual_info_classif, k=3);
# Fit and Transform

x_fs3 = FS3.fit_transform(x, y.values.ravel());
# The chosen 3 variables

fs3_cols = x.columns[FS3.get_support()]

fs3_cols
# Import our favorite library

from sklearn.cluster import KMeans
# Create our data variable

train = x[fs2_cols]
# Elbow Method for K-Means

distortions = []

for i in range(1, 11):

    km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, tol=1e-04, random_state=1206)

    km.fit(train)

    distortions.append(km.inertia_)
# plot Elbow

plt.plot(range(1, 11), distortions, marker='o')

plt.xlabel('Number of clusters')

plt.ylabel('Distortion')

plt.show()
# Creating KMeans model with 3 clusters as was propose

km3 = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300, tol=1e-04, random_state=1206)
# Fit/Predict KMeans 3 Cluster

pred2D = km3.fit_predict(train);
# Create auxiliar dataframe for plot

data = pd.concat([train, y], axis=1)
# Create a column for predicted clusters from KMeans

for i in range(0,len(data)):

    if pred2D[i] == 0:

        data.loc[i,'PREDICT'] = 'Cluster 1'

    elif pred2D[i] == 1:

        data.loc[i,'PREDICT'] = 'Cluster 2'

    elif pred2D[i] == 2:

        data.loc[i,'PREDICT'] = 'Cluster 3'
# Plot 2D

fig, ax = plt.subplots(1,2, figsize=(20,10));

sns.scatterplot(x='IBGE_RES_POP_ESTR', y='IBGE_DU_URBAN', hue="PREDICT", data=data, ax=ax[0], sizes=0.1);

sns.scatterplot(x='IBGE_RES_POP_ESTR', y='IBGE_DU_URBAN', hue="IDHM_CAT2", data=data, ax=ax[1]);
# Create our data variable

train = x[fs3_cols]
# Elbow Method for K-Means

distortions = []

for i in range(1, 11):

    km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, tol=1e-04, random_state=1206)

    km.fit(train)

    distortions.append(km.inertia_)

# plot Elbow

plt.plot(range(1, 11), distortions, marker='o')

plt.xlabel('Number of clusters')

plt.ylabel('Distortion')

plt.show()
# Creating KMeans model with 3 clusters as was propose

km3 = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300, tol=1e-04, random_state=1206)
# Fit/Predict KMeans 3 Cluster

pred3D = km3.fit_predict(train);
# Create auxiliar dataframe for plot

data = pd.concat([train, y], axis=1)
# Create a column for predicted clusters from KMeans

for i in range(0,len(data)):

    if pred2D[i] == 0:

        data.loc[i,'PREDICT'] = 'Cluster 1'

        #data.loc[i,'COLOR'] = 'blue'

    elif pred2D[i] == 1:

        data.loc[i,'PREDICT'] = 'Cluster 2'

        #data.loc[i,'COLOR'] = 'green'

    elif pred2D[i] == 2:

        data.loc[i,'PREDICT'] = 'Cluster 3'

        #data.loc[i,'COLOR'] = 'orange'
# Import library for 3D plot

import plotly.express as px
# Plot 3D - Original Variables Cluster

px.scatter_3d(data, x='IBGE_DU_URBAN', y='IBGE_RES_POP_ESTR', z='IBGE_60+', color='IDHM_CAT2')
# Plot 3D - Original Variables Cluster

px.scatter_3d(data, x='IBGE_DU_URBAN', y='IBGE_RES_POP_ESTR', z='IBGE_60+', color='PREDICT')



## other way to plot

# Library for 3D

#from mpl_toolkits.mplot3d import Axes3D

# Plot 3D

#plotting = plt.figure(figsize=(15,15)).gca(projection='3d');

#plotting.scatter(data['IBGE_DU_URBAN'], data['IBGE_RES_POP_ESTR'], data['IBGE_60+'],c=data['COLOR']);

#plotting.set_xlabel('IBGE_DU_URBAN');

#plotting.set_ylabel('IBGE_RES_POP_ESTR');

#plotting.set_zlabel('IBGE_60+');
# Import our model

from sklearn.decomposition import PCA
# Create model

pca = PCA(n_components=2)
# Fit and Transform X to PC dimension

pc = pca.fit_transform(x)
# See explained variance

pca.explained_variance_ratio_
# Create a Dataframe for PC with target var

xPC = pd.DataFrame(data = pc, columns = ['PC1', 'PC2'])

# Join with target categories

xPC = pd.concat([xPC, y], axis = 1)
# Plot 2D

plt.figure(figsize=(20,10));

sns.scatterplot(x="PC2", y="PC1", hue="IDHM_CAT2", data=xPC);
# Create model

pca = PCA(n_components=3)
# Fit and Transform X to PC dimension

pc = pca.fit_transform(x)
# Create a Dataframe for PC with target var

xPC = pd.DataFrame(data = pc, columns = ['PC1', 'PC2', 'PC3'])

# Join with target categories

xPC = pd.concat([xPC, y], axis = 1)
# Plot 3D

px.scatter_3d(xPC, x='PC1', y='PC2', z='PC3', color='IDHM_CAT2')
# See explained variance

pca.explained_variance_ratio_
# Import library for TSNE

from sklearn.manifold import TSNE
# Create tSNE model

tsne = TSNE(n_components=2)
# Fit and Transform x

x_emb = tsne.fit_transform(x)
# Turn X embeddeb into a dataframe to a easy plot

x_emb = pd.DataFrame(data = x_emb, columns = ['XE1', 'XE2'])

# Join with target categories

x_emb = pd.concat([x_emb, y], axis = 1)
# Plot 2D data

sns.scatterplot(x='XE1', y='XE2', hue="IDHM_CAT2", data=x_emb);
# Create tSNE model

tsne = TSNE(n_components=3)
# Fit and Transform x

x_emb = tsne.fit_transform(x)
# Turn X embedded into a dataframe to an easy plot

x_emb = pd.DataFrame(data = x_emb, columns = ['XE1', 'XE2', 'XE3'])

# Join with target categories

x_emb = pd.concat([x_emb, y], axis = 1)
# Plot 3D

px.scatter_3d(x_emb, x='XE1', y='XE2', z='XE3', color='IDHM_CAT2')