import numpy

import pandas

import sklearn

import scipy

import collections

import matplotlib

import seaborn

import sys
modules = list(set(sys.modules) & set(globals()))

for module_name in modules:

    module = sys.modules[module_name]

    print(module_name, getattr(module, '__version__', 'unknown'))
import numpy as np

import pandas as pd



from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score



from scipy.cluster.hierarchy import dendrogram, linkage

from scipy.cluster.hierarchy import fcluster

from scipy.cluster.hierarchy import cophenet

from scipy.spatial.distance import pdist

from scipy import stats

from scipy.stats import norm, skew



from collections import Counter



import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline





import warnings

warnings.filterwarnings('ignore')
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
data = pd.read_csv('../input/faulty-steel-plates/faults.csv')
data.head()
data.shape
data.describe()
targets = data.iloc[:, 27:35]

data.drop(targets.columns, axis=1, inplace=True)

data['Target'] = targets.idxmax(1)

data.head()
origina_data = data.copy()
target_counts= data['Target'].value_counts()



fig, ax = plt.subplots(1, 2, figsize=(15,7))

target_counts_barplot = sns.barplot(x = target_counts.index,y = target_counts.values, ax = ax[0])

target_counts_barplot.set_ylabel('Number of classes in the dataset')



colors = ['#8d99ae','#ffe066', '#f77f00','#348aa7','#bce784','#ffcc99',  '#f25f5c']

target_counts.plot.pie(autopct="%1.1f%%", ax=ax[1], colors=colors)

sns.pairplot(data, hue='Target')
data['TypeOfSteel_A300'] = data['TypeOfSteel_A300'].astype('category',copy=False)

data['TypeOfSteel_A400'] = data['TypeOfSteel_A400'].astype('category',copy=False)

data['Outside_Global_Index'] = data['Outside_Global_Index'].astype('category',copy=False)
plt.figure(figsize=(20,15))

sns.heatmap(data.corr(), cmap='seismic')
sns.regplot(x='X_Minimum', y='X_Maximum', data = data, scatter = True)
sns.regplot(x='Pixels_Areas', y='Sum_of_Luminosity', data = data, scatter = True)
sns.regplot(x='Maximum_of_Luminosity', y='Luminosity_Index', data = data, scatter = True)
sns.regplot(x='Edges_Y_Index', y='Log_X_Index', data = data, scatter = True)
numeric_features = data.dtypes[data.dtypes != "object"].index



skewed_features = data[numeric_features].apply(lambda x: skew(x)).sort_values(ascending=False)



skewed_features_df = pd.DataFrame(skewed_features, columns={'Skew'})

skewed_features_df.head(10)
skewed_features_df.tail(10)
skewed_features_df.drop(['TypeOfSteel_A400','TypeOfSteel_A300', 'Outside_Global_Index'], inplace=True)
sns.distplot(data['Sum_of_Luminosity'])
skewed_features_df = skewed_features_df[abs(skewed_features_df) > 0.75]



from scipy.special import boxcox1p

lam = 0.15

cols = skewed_features_df.index



for c in cols:

    data[c] = boxcox1p(data[c], lam)
sns.distplot(data['Sum_of_Luminosity'])
features = data.drop('Target', axis=1)

target = data['Target']



scaler = StandardScaler()

features_scaled = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)
data_boxplot = features_scaled.boxplot(return_type='dict', vert=False, figsize=(20,20))
features_scaled[features_scaled['Pixels_Areas']>4]
features_scaled[features_scaled['Sum_of_Luminosity']>4]
features_scaled[features_scaled['X_Perimeter']>4]
features_scaled[features_scaled['Y_Perimeter']>4]
data_scaled = features_scaled.copy()

data_scaled['Target'] = target



data_scaled['Target'] = pd.Categorical(data_scaled['Target'])

data_scaled['Target_Code'] = data_scaled.Target.cat.codes
plt.figure(figsize=(8,5))

sns.boxplot(x='Target', y='X_Maximum', data=data_scaled)
plt.figure(figsize=(8,5))

sns.boxplot(x='Target', y='Steel_Plate_Thickness', data=data_scaled)
plt.figure(figsize=(8,5))

sns.boxplot(x='Target', y='Luminosity_Index', data=data_scaled)
plt.figure(figsize=(8,5))

sns.boxplot(x='Target', y='Square_Index', data=data_scaled)
plt.figure(figsize=(8,5))

sns.boxplot(x='Target', y='Edges_Index', data=data_scaled)
plt.figure(figsize=(8,5))

sns.boxplot(x='Target', y='LogOfAreas', data=data_scaled)
plt.figure(figsize=(8,5))

sns.boxplot(x='Target', y='Y_Maximum', data=data_scaled)
plt.figure(figsize=(8,5))

sns.boxplot(x='Target', y='Orientation_Index', data=data_scaled)
plt.figure(figsize=(8,5))

sns.boxplot(x='Target', y='Minimum_of_Luminosity', data=data_scaled)
plt.figure(figsize=(8,5))

sns.boxplot(x='Target', y='Length_of_Conveyer', data=data_scaled)
dbscan_model = DBSCAN(eps=3.3, min_samples=7).fit(features_scaled)
print(Counter(dbscan_model.labels_))
outliers = features_scaled[dbscan_model.labels_ == -1]

outliers.shape
features_scaled.drop(outliers.index, axis=0, inplace=True)

target.drop(outliers.index, axis=0, inplace=True)

data_scaled.drop(outliers.index, axis=0, inplace=True)

features_scaled.shape
pca = PCA(random_state=101)

features_pca = pca.fit_transform(features_scaled.values)

pd.DataFrame(pca.explained_variance_ratio_, columns=['Explained Variance Ratio'])
pca.explained_variance_ratio_[0:15].sum()
pca_components = pd.DataFrame(pca.components_, columns= features.columns)

plt.figure(figsize=(20,20))

sns.heatmap(pca_components, cmap='seismic')
def pca_dataset(features, n_components):

    

    pca_n = PCA(n_components=n_components, random_state=101)

    features_pca_n = pca_n.fit_transform(features)

    

    column_pca = []

    for i in range(0,n_components):

        column_pca.append('Component'+np.str(i))

    return pd.DataFrame(features_pca_n, columns=column_pca)
data_pca15 = pca_dataset(features_scaled, n_components=15)

data_pca15['Target'] = target
sns.pairplot(data_pca15, hue='Target')
pca.explained_variance_ratio_[0:5].sum()
data_pca5 = pca_dataset(features_scaled, n_components=5)

data_pca5['Target'] = target
data_pca15['Target'] = pd.Categorical(data_pca15['Target'])

data_pca15['Target_Code'] = data_pca15.Target.cat.codes



data_pca5['Target'] = pd.Categorical(data_pca5['Target'])

data_pca5['Target_Code'] = data_pca5.Target.cat.codes
kmeans_model = KMeans(n_clusters=7, random_state=54)

kmeans_model.fit(features_scaled)
kmeans_labels = np.choose(kmeans_model.labels_, [0,1,2,3,4,5,6]).astype(np.int64)

data_scaled['kmeans_labels'] = kmeans_labels
color_themes = {0:'#8d99ae',1:'#ffe066', 2:'#f77f00',3:'#348aa7',4:'#bce784',5:'#ffcc99',  6:'#f25f5c'}





sns.lmplot(x='Orientation_Index', y='Log_X_Index', data=data_scaled, fit_reg=False, hue='Target', col='Target', size=8)

plt.title("Ground Truth Classification")



sns.lmplot(x='Orientation_Index', y='Log_X_Index', data=data_scaled,  fit_reg=False, hue='kmeans_labels', col='kmeans_labels',size=8)

plt.title("KMean Clustering")
print(classification_report(data_scaled['Target_Code'], kmeans_labels))
kmeans_model_pca15 = KMeans(n_clusters=7, random_state=54)

kmeans_model_pca15.fit(data_pca15.drop(['Target','Target_Code'], axis=1))
kmeans_labels_pca15 = np.choose(kmeans_model.labels_, [0,1,2,3,4,5,6]).astype(np.int64)

data_pca15['kmeans_labels'] = kmeans_labels_pca15
sns.lmplot(x='Component0', y='Component1', data=data_pca15, fit_reg=False, hue='Target', col='Target', size=8)

plt.title("Ground Truth Classification")



sns.lmplot(x='Component0', y='Component1', data=data_pca15,  fit_reg=False, hue='kmeans_labels', col='kmeans_labels',size=8)

plt.title("KMean Clustering")
print(classification_report(data_pca15['Target_Code'], kmeans_model_pca15.labels_))
kmeans_model_pca5 = KMeans(n_clusters=7, random_state=54)

kmeans_model_pca5.fit(data_pca5.drop(['Target','Target_Code'], axis=1))
print(classification_report(data_pca5['Target_Code'], kmeans_model_pca15.labels_))
original_features = origina_data.drop(['Target'], axis=1).copy()

origina_data['Target'] = pd.Categorical(origina_data['Target'])

origina_data['Target_Code'] = origina_data.Target.cat.codes
linkage_model = linkage(original_features, method='ward')

dendrogram(linkage_model, truncate_mode='lastp', p=12, leaf_rotation=45, leaf_font_size=12, show_contracted=True)

plt.title('Truncated Hierarchical Clustering Dendrogram')

plt.xlabel('Cluster Size')

plt.ylabel('Distance')



plt.axhline(y=0.4*10**(8))

plt.axhline(y=0.2*10**(8))
k = 7

h_clustering = AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='ward')

h_clustering.fit(original_features)



accuracy_score(origina_data['Target_Code'], h_clustering.labels_)
h_clustering = AgglomerativeClustering(n_clusters=k, affinity='manhattan', linkage='complete' )

h_clustering.fit(original_features)



accuracy_score(origina_data['Target_Code'], h_clustering.labels_)
h_clustering = AgglomerativeClustering(n_clusters=k, affinity='manhattan', linkage='average')

h_clustering.fit(original_features)



accuracy_score(origina_data['Target_Code'], h_clustering.labels_)
k = 7

h_clustering_pca5 = AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='ward' )

h_clustering_pca5.fit(data_pca5.drop(['Target','Target_Code'], axis=1))



accuracy_score(data_pca5['Target_Code'], h_clustering_pca5.labels_)