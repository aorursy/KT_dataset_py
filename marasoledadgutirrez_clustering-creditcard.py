# data management
import pandas as pd
import numpy as np

# visualization
from pylab import*
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# preprocessing
import sklearn as sk
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer

# clusters models
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, DBSCAN
from sklearn import metrics
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage
data = pd.read_csv("../input/ccdata/CC GENERAL.csv")
data.shape
data.head(3)
features = data.columns[1:]
data.info()
data[features].describe()
data.nunique()
data.isna().sum()
print(data[data.CREDIT_LIMIT.isna()].shape[0],' clientes')
print("{0:.2f}%".format(100*data[data.CREDIT_LIMIT.isna()].shape[0]/data.shape[0]))
data[data.CREDIT_LIMIT.isna()]
data.CREDIT_LIMIT.describe()
print('Customers with zero credit limit:' , data[data.CREDIT_LIMIT==0].shape[0])
data_aux = data[(data.PURCHASES_TRX==0)&(data.CASH_ADVANCE_TRX>0)][['CASH_ADVANCE','CASH_ADVANCE_TRX','CREDIT_LIMIT']]
print(data_aux.describe())
data_aux.head()
print(data[data.MINIMUM_PAYMENTS.isna()].shape[0],' clientes')
print("{0:.2f}%".format(100*data[data.MINIMUM_PAYMENTS.isna()].shape[0]/data.shape[0]))
data[data.MINIMUM_PAYMENTS.isna()].head(7)
data[(data.PAYMENTS==0)].shape[0] == data[(data.PAYMENTS==0)&(data.MINIMUM_PAYMENTS.isna())].shape[0]
data[(data.MINIMUM_PAYMENTS.isna())&(data.PRC_FULL_PAYMENT==0)].shape[0] == data[data.MINIMUM_PAYMENTS.isna()].shape[0]
data.MINIMUM_PAYMENTS.describe()
def detect_col_outliers(ls_data):
     # z_score and filter

    mean = np.mean(ls_data)
    std = np.std(ls_data)
   
    return [i for i in ls_data if np.abs(i-mean) > 4*std]
features_outliers = ['BALANCE','PURCHASES','ONEOFF_PURCHASES','INSTALLMENTS_PURCHASES','CASH_ADVANCE','CASH_ADVANCE_TRX','PURCHASES_TRX','CREDIT_LIMIT','PAYMENTS','MINIMUM_PAYMENTS']
for name_col in features_outliers:
    rtdo = detect_col_outliers(data[name_col])
    print('-'*50)
    print(name_col)
    print('# values outlier: ', len(rtdo))
    print('{0:.2f}% of the total data'.format(100*len(rtdo)/data.shape[0]))
plt.figure(figsize=(15,10))
sns.boxplot(data=data[features])
plt.xticks(rotation=90)
nr_rows = len(features_outliers)
nr_cols = 3

fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*3.5,nr_rows*3))

for r, col in enumerate(features_outliers):
    sns.distplot(data[col], ax = axs[r][0]).set_title('Original')
    sns.distplot(np.sqrt(data[col].tolist()), ax = axs[r][1]).set_title("Root Square")
    sns.distplot(np.log1p(data[col]), ax = axs[r][2]).set_title('log(1+x)')
plt.tight_layout()    
plt.show()  
int_cols = data[features].select_dtypes(include=['int']).columns
int_cols
for col in int_cols:
    print(data[col].value_counts().sort_values(ascending=False))
    print('-'*30)
data[int_cols].hist(figsize=(15,8))
plt.tight_layout()
#Using Pearson Correlation
plt.figure(figsize=(12,10))
corr_m = data[features].corr()
sns.heatmap(corr_m, annot=True, cmap=plt.cm.Reds).set_title('Correlation Matrix')
plt.show()
cor_purchases = abs(corr_m["PURCHASES"])
cor_purchases[cor_purchases>0.5].sort_values(ascending=False)
print('{0:.2f}%'.format(100*sum(data.PURCHASES == data.ONEOFF_PURCHASES + data.INSTALLMENTS_PURCHASES)/data.shape[0]))
data[data.PURCHASES != data.ONEOFF_PURCHASES + data.INSTALLMENTS_PURCHASES].head()
sns.pairplot(data[['PURCHASES','ONEOFF_PURCHASES','INSTALLMENTS_PURCHASES']],
             markers="+",
             kind='reg',
             diag_kind=None, 
             height=4)
sns.pairplot(data[['CASH_ADVANCE_FREQUENCY','CASH_ADVANCE_TRX']],
             markers="+",
             kind='reg',
             height=4)
features = data.columns[1:]
features_group1 = ['BALANCE','ONEOFF_PURCHASES','INSTALLMENTS_PURCHASES','CASH_ADVANCE','CASH_ADVANCE_TRX','PURCHASES_TRX','PAYMENTS','CREDIT_LIMIT','MINIMUM_PAYMENTS']
features_group2 = list(set(features)-set(features_group1))
# using median in columns with outliers 
g1_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('log', FunctionTransformer(np.log1p)),
    #('scaler', MinMaxScaler(feature_range=(0, 1)))
    ('scaler', StandardScaler())
    ])

# using median in columns without outliers 
g2_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
    ])

preprocessor = ColumnTransformer(
    transformers=[
        ('group1', g1_transformer, features_group1),
        ('group2', g2_transformer, features_group2),
        ])
preprocessor.fit(data) 
np_data = preprocessor.transform(data) 
print(np_data[np.isnan(np_data)])
df_data = pd.DataFrame(np_data, columns=features_group1+features_group2)
print(df_data.isna().sum())
print(df_data.shape)
df_data.head(6)
#to check StandardScaler
df_data.describe()
# to check outliers
plt.figure(figsize=(15,10))
sns.boxplot(data=df_data)
plt.xticks(rotation=90)
data_range = data.copy()
columns=['BALANCE', 'PURCHASES', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS']

for c in columns:
    
    Range=c+'_RANGE'
    data_range[Range]=0        
    data_range.loc[((data[c]>0)&(data[c]<=500)),Range]=1
    data_range.loc[((data[c]>500)&(data[c]<=1000)),Range]=2
    data_range.loc[((data[c]>1000)&(data[c]<=3000)),Range]=3
    data_range.loc[((data[c]>3000)&(data[c]<=5000)),Range]=4
    data_range.loc[((data[c]>5000)&(data[c]<=10000)),Range]=5
    data_range.loc[((data[c]>10000)),Range]=6
columns=['BALANCE_FREQUENCY', 'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_FREQUENCY', 'PRC_FULL_PAYMENT']

for c in columns:  

    Range=c+'_RANGE'
    data_range[Range]=0
    for i in range(10):
        data_range.loc[((data[c]>i*0.1)&(data[c]<=(i+1)*0.1)), Range]=i+1
columns=['PURCHASES_TRX', 'CASH_ADVANCE_TRX']  

for c in columns:
    
    Range=c+'_RANGE'
    data_range[Range]=0
    data_range.loc[((data[c]>0)&(data[c]<=5)),Range]=1
    data_range.loc[((data[c]>5)&(data[c]<=10)),Range]=2
    data_range.loc[((data[c]>10)&(data[c]<=15)),Range]=3
    data_range.loc[((data[c]>15)&(data[c]<=20)),Range]=4
    data_range.loc[((data[c]>20)&(data[c]<=30)),Range]=5
    data_range.loc[((data[c]>30)&(data[c]<=50)),Range]=6
    data_range.loc[((data[c]>50)&(data[c]<=100)),Range]=7
    data_range.loc[((data[c]>100)),Range]=8
data_range.drop(['CUST_ID', 'BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES',
       'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE',
       'PURCHASES_FREQUENCY',  'ONEOFF_PURCHASES_FREQUENCY',
       'PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_FREQUENCY',
       'CASH_ADVANCE_TRX', 'PURCHASES_TRX', 'CREDIT_LIMIT', 'PAYMENTS',
       'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT' ], axis=1, inplace=True)
len(data.columns), len(data_range.columns)
data_range.head()
data_range.describe()
plt.figure(figsize=(15,10))
sns.boxplot(data=data_range)
plt.xticks(rotation=90)
features_group3 = ['INSTALLMENTS_PURCHASES_RANGE','MINIMUM_PAYMENTS_RANGE','ONEOFF_PURCHASES_FREQUENCY_RANGE','CASH_ADVANCE_FREQUENCY_RANGE','PRC_FULL_PAYMENT_RANGE','CASH_ADVANCE_TRX_RANGE']
features_group4 = list(set(data_range.columns)-set(features_group3))
# using median in columns with outliers 
g1_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('log', FunctionTransformer(np.log1p)),
    #('scaler', MinMaxScaler(feature_range=(0, 1)))
    ('scaler', StandardScaler())
    ])

# using median in columns without outliers 
g2_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
    ])

preprocessor2 = ColumnTransformer(
    transformers=[
        ('group1', g1_transformer, features_group3),
        ('group2', g2_transformer, features_group4),
        ])
data_range.columns
preprocessor2.fit(data_range) 
np_data_range = preprocessor2.transform(data_range) 
print(np_data_range[np.isnan(np_data_range)])
df_data2 = pd.DataFrame(np_data_range, columns=features_group3+features_group4)
print(df_data2.isna().sum())
print(df_data2.shape)
df_data2.head(6)
df_data.describe()
plt.figure(figsize=(15,10))
sns.boxplot(data=df_data)
plt.xticks(rotation=90)
pca = PCA(n_components=2)
pca.fit(np_data)
data_pca = pca.transform(np_data)
plt.figure(figsize=(8,6))
plt.scatter(np_data[:,0],np_data[:,1])
plt.xlabel('First principal component')
plt.ylabel('Second Principal Component')
print(pca.noise_variance_)
print(pca.explained_variance_ratio_)
Sum_of_squared_distances = []
K = range(1, 20)
for k in K:
    km = KMeans(n_clusters=k, 
                init='k-means++',
                max_iter=400, 
                n_init=80, 
                random_state=0).fit(np_data)
    Sum_of_squared_distances.append(km.inertia_)

plt.figure(figsize=(10,10))
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()
silhouette_scores = [] 
K = range(2, 20)

for k in K:
    km = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=45).fit_predict(np_data)
    scr = silhouette_score(np_data, km)
    silhouette_scores.append(scr)
    print("For n_clusters =", k, "The average silhouette_score is :", scr)
plt.plot(K, silhouette_scores, 'bx-')
plt.xlabel('k')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Method For Optimal k')
plt.show()


K = range(2,10)

for k in K:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(np_data) + (k + 1) * 10])

    clusterer = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=45)
    cluster_labels = clusterer.fit_predict(np_data)

    silhouette_avg = silhouette_score(np_data, cluster_labels)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(np_data, cluster_labels)

    y_lower = 10
    for i in range(k):
        # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / k)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])


    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / k)
    pca = PCA(n_components=2)
    pca.fit(np_data)
    X = pca.transform(np_data)

    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # Labeling the clusters
    pca_centers = pca.transform(clusterer.cluster_centers_)
    # Draw white circles at cluster centers
    ax2.scatter(pca_centers[:, 0], pca_centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(pca_centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st principal feature")
    ax2.set_ylabel("Feature space for the 2nd principal feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % k),
                 fontsize=14, fontweight='bold')

plt.show()
km = KMeans(n_clusters=6, 
            init='k-means++',
            max_iter=400, 
            n_init=80, 
            random_state=0)

km_pipe = Pipeline(steps=[('preprocessor', preprocessor),
                          ('km', km)])

km_pipe.fit(data)
labels = km.labels_
clusters = pd.concat([data, pd.DataFrame({'CLUSTER':labels})], axis=1)
clusters.head()
clusters.CLUSTER.value_counts()
clusters.CLUSTER.hist(figsize=(10, 8))
plt.tight_layout()
# save clusters to csv
clusters.to_csv('Clusters_CreditCards_Kmeans.csv')
for c in clusters:
    grid= sns.FacetGrid(clusters, col='CLUSTER')
    grid.map(plt.hist, c)
clusters.groupby(['CLUSTER']).mean()
dist = 1 - cosine_similarity(np_data)

pca = PCA(2)
pca.fit(dist)
X_PCA = pca.transform(dist)
X_PCA.shape
x, y = X_PCA[:, 0], X_PCA[:, 1]

colors = {0: 'red',
          1: 'blue',
          2: 'green', 
          3: 'yellow', 
          4: 'orange',  
          5:'purple'}

names = {0: 'high level of income and high credit limit who take cash in advance', 
         1: 'low level of income. Not Frequent purchases', 
         2: 'who purchases mostly in installments', 
         3: 'They purchase mostly in one-go with a high frequency. the percent of full payment paid is low (debtors)', 
         4: 'do not spend much money and who accept large amounts of cash advances but not frequently',
         5: 'High spenders who take more cash in advance'}
  
df = pd.DataFrame({'x': x, 'y':y, 'label':labels}) 
groups = df.groupby('label')

fig, ax = plt.subplots(figsize=(20, 13)) 

for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=5,
            color=colors[name],label=names[name], mec='none')
    ax.set_aspect('auto')
    ax.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
    ax.tick_params(axis= 'y',which='both',left='off',top='off',labelleft='off')
    
ax.legend()
ax.set_title("Customers Segmentation based on their Credit Card usage bhaviour.")
plt.show()
preprocessor.fit(data) 
np_data = preprocessor.transform(data) 
siliuette_list_hierarchical = []
for cluster in range(2,10):
    for linkage_method in ['ward', 'average','single']:
        agglomerative = AgglomerativeClustering(linkage=linkage_method, affinity='euclidean',n_clusters=cluster).fit_predict(np_data)
        sil_score = metrics.silhouette_score(np_data, agglomerative, metric='euclidean')
        siliuette_list_hierarchical.append((cluster, sil_score, linkage_method))
        
df_hierarchical = pd.DataFrame(siliuette_list_hierarchical, columns=['cluster', 'sil_score','linkage_method'])
df_hierarchical.sort_values('sil_score', ascending=False)
Z_avg = linkage(np_data, 'average')

plt.figure(figsize=(15,10))
dendrogram(Z_avg, leaf_rotation=90, p=5, color_threshold=20, leaf_font_size=10, truncate_mode='level')
plt.axhline(y=125, color='r', linestyle='--')
plt.show()
Z_ward = linkage(np_data, 'ward')

plt.figure(figsize=(15,10))
dendrogram(Z_ward, leaf_rotation=90, p=5, color_threshold=20, leaf_font_size=10, truncate_mode='level')
plt.axhline(y=125, color='r', linestyle='--')
plt.show()
Z_ward = linkage(np_data, 'single')

plt.figure(figsize=(15,10))
dendrogram(Z_ward, leaf_rotation=90, p=15, color_threshold=20, leaf_font_size=10, truncate_mode='level')
plt.axhline(y=125, color='r', linestyle='--')
plt.show()
hierarchical = AgglomerativeClustering(n_clusters=2, linkage='average')
pipe_hierar = Pipeline(steps=[
                              ('preprocessor', preprocessor),
                              ('hierarchical', hierarchical)]
                       )

pipe_hierar.fit(data)
clusters_hierar = pd.concat([data, pd.DataFrame({'CLUSTER':hierarchical.labels_})], axis=1)
clusters_hierar.head()
clusters_hierar.to_csv('Clusters_CreditCard_Hierarchical.csv')
clusters_hierar.groupby('CLUSTER').mean()
clusters_hierar.CLUSTER.value_counts()
