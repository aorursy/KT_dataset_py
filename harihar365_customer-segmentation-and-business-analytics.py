import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sc
import matplotlib as mpl
import itertools
import warnings
warnings.filterwarnings('ignore')
data = pd.read_csv('../input/ulabox_orders_with_categories_partials_2017.csv')
data.head()
data.describe()
data[data['discount%']<0].sort_values(by='discount%', ascending=True).head(10)
indices = [56,2459,908,23632,1803,218,592,349]
data.iloc[indices, :]
df = data.drop(['customer', 'order', 'hour'], axis=1)
frame = data
from sklearn.decomposition import PCA
pca = PCA(n_components=11)
pca.fit(df.values)
def pca_2d_plot(pca, df):
    fig = plt.figure(figsize=(10,10))
    transformed_data = pca.transform(df.values)
    data = pd.DataFrame(transformed_data, columns=['dim'+str(i) for i in range(1,12)])
    sns.lmplot(x='dim1', y='dim2', data=data, size=12, fit_reg=False, scatter_kws={'s':8});
    sns.lmplot(x='dim3', y='dim4', data=data, size=12, fit_reg=False, scatter_kws={'s':8});
    plt.show()
pca_2d_plot(pca, df)
figure = plt.figure(figsize=(20,20))
sns.pairplot(df);
plt.show()
fig = plt.figure(figsize=(16,12))
sns.distplot(df['total_items']);
plt.show()
df['total_items'] = np.log(df['total_items'])
fig = plt.figure(figsize=(16,12))
sns.distplot(df['total_items']);
plt.show()
def turkey_outlier_detector(df, cols=None):
    if cols  is None:
        cols = [str(s) for s in df.describe().columns]
        
    q1 = {}
    q3 = {}
    iqd = {}
    r_limit = {}
    l_limit = {}
    outlier_count = {}
    outlier_indices = {}
    for col in cols:
        q1[col] = np.percentile(df[col].values, 25)
        q3[col] = np.percentile(df[col].values, 75)
        iqd[col] = q3[col] - q1[col]
        r_limit[col] = q3[col] + 1.5*iqd[col]
        l_limit[col] = q1[col] - 1.5*iqd[col]
        data_outlier = df[~((df[col]<r_limit[col]).multiply(df[col]>l_limit[col]))]
        outlier_count[col] = data_outlier.shape[0]
        outlier_indices[col] = data_outlier.index
        
    for col in cols:
        print('_'*25)
        print(col+'-'*8+'>'+str(outlier_count[col]))
        
    return outlier_indices
outlier_indices = turkey_outlier_detector(df)
df.drop(outlier_indices['total_items'], inplace=True)
frame.drop(outlier_indices['total_items'], inplace=True)
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(df.values)
clusters = range(3,31)
inertia = []
for n in clusters:
    kmeans = KMeans(n_clusters=n)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)
    
fig, ax = plt.subplots(figsize=(10,8))
ax.plot(clusters, inertia);
plt.show()
def plot_silhoutte_score(X, max_clusters=20):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    num_clusters = range(2,max_clusters+1)
    sil_score = []
    for n in num_clusters:
        kmeans = KMeans(n_clusters=n)
        kmeans.fit(X)
        preds = kmeans.predict(X)
        sil_score.append(silhouette_score(X, preds))
        
    fig, ax = plt.subplots(figsize=(12,12))
    ax.plot(num_clusters, sil_score)
    plt.show()
plot_silhoutte_score(X,30)
def under_partition_measure(X, k_max):
    from sklearn.cluster import KMeans
    ks = range(1,k_max+1)
    UPM = []
    for k in ks:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        UPM.append(kmeans.inertia_)
    return UPM
def over_partition_measure(X, k_max):
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import  pairwise_distances
    ks = range(1,k_max+1)
    OPM = []
    for k in ks:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        centers = kmeans.cluster_centers_
        d_min = np.inf
        for pair in list(itertools.combinations(centers, 2)):
            d = pairwise_distances(pair[0].reshape(1,-1), pair[1].reshape(1,-1), metric='euclidean')
            if d<d_min:
                d_min = d
        OPM.append(k/d_min)
    return OPM
def validity_index(X, k_max):
    UPM = under_partition_measure(X, k_max)
    OPM = over_partition_measure(X, k_max)
    UPM_min = np.min(UPM)
    OPM_min = np.min(OPM)
    UPM_max = np.max(UPM)
    OPM_max = np.max(OPM)
    norm_UPM = []
    norm_OPM = []
    for i in range(k_max):
        norm_UPM.append((UPM[i]-UPM_min)/(UPM_max-UPM_min))
        norm_OPM.append((OPM[i]-OPM_min)/(OPM_max-OPM_min))
        
    validity_index = np.array(norm_UPM)+np.array(norm_OPM)
    fig, ax = plt.subplots(figsize=(12,12))
    ax.plot(range(1,k_max+1), validity_index)
    return validity_index
_ = validity_index(X, 30)
kmeans_10 = KMeans(n_clusters=10, random_state=42)
kmeans_10.fit(X)
frame['labels'] = kmeans_10.predict(X)
frame[frame['labels']==0].head(10)
frame[frame['labels']==0].describe()
frame.loc[frame['labels']==0, 'class'] = 'drink_buyers'
frame[frame['labels']==1].head(10)
frame[frame['labels']==1].describe()
frame.loc[frame['labels']==1, 'class'] = 'loyals_fresh'
frame[frame['labels']==2].head(10)
frame[frame['labels']==2].describe()
frame.loc[frame['labels']==2, 'class'] = 'loyals_grocery'
frame[frame['labels']==3].head(10)
frame[frame['labels']==3].describe()
frame.loc[frame['labels']==3, 'class'] = 'beauty_concious'
frame[frame['labels']==4].head(10)
frame[frame['labels']==4].describe()
frame.loc[frame['labels']==4, 'class'] = 'health_concious'
frame[frame['labels']==5].head(10)
frame[frame['labels']==5].describe()
frame.loc[frame['labels']==5, 'class'] = 'loyals'
frame[frame['labels']==6].head(10)
frame[frame['labels']==6].describe()
frame.loc[frame['labels']==6, 'class'] = 'grocery_shoppers'
frame[frame['labels']==7].head(10)
frame[frame['labels']==7].describe()
frame.loc[frame['labels']==7, 'class'] = 'home_decorators'
frame[frame['labels']==8].head(10)
frame[frame['labels']==8].describe()
frame.loc[frame['labels']==8, 'class'] = 'pet_lovers'
frame[frame['labels']==9].head(10)
frame[frame['labels']==9].describe()
frame.loc[frame['labels']==9, 'class'] = 'new_parents'
def pca_2d_plot_labels(pca, df, frame):
    plt.figure(figsize=(18,18));
    transformed_data = pca.transform(df.values)
    data = pd.DataFrame({'dim1':transformed_data[:,0], 'dim2':transformed_data[:,1], 'labels':frame['class'].values})
    sns.lmplot(x='dim1',y='dim2',hue='labels',data=data, fit_reg=False, size=16);
    data1 = pd.DataFrame({'dim2':transformed_data[:,1], 'dim3':transformed_data[:,2], 'labels':frame['class'].values})
    sns.lmplot(x='dim2',y='dim3',hue='labels',data=data1, fit_reg=False, size=16);
    plt.show()
pca_2d_plot_labels(pca, df, frame)
frame.groupby('class')['total_items'].describe()
frame.groupby('class')['discount%'].describe()
frame['class'].value_counts().sort_values(ascending=False)
plt.figure(figsize=(9,9))
frame['class'].value_counts().sort_values(ascending=False).plot.pie(autopct='%1.0f%%', labels=list(frame['class'].value_counts().sort_values(ascending=False).index))
plt.show()
plt.figure(figsize=(9,9))
frame[frame['discount%']<0]['class'].value_counts().sort_values(ascending=False).plot.pie(autopct='%1.0f%%', labels=frame[frame['discount%']<0]['class'].value_counts().sort_values(ascending=False).index)
plt.show()
frame[(frame['discount%']<0).multiply(frame['class']!='drink_buyers')].describe()
frame[frame['discount%']<0].shape[0]
