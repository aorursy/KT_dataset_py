from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.decomposition import FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import silhouette_score

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


%matplotlib inline
data=pd.read_csv("../input/sgemm/sgemm_product.csv")
data.head()
#take average of 4 run
data["run_avg"]=np.mean(data.iloc[:,14:18],axis=1)

mean_run=np.mean(data["run_avg"])
print(mean_run)

#Binary Classification run_avg>mean_run
data["run_class"]=np.where(data['run_avg']>=mean_run, 1, 0)
data.groupby("run_class").size()

data.describe()
sgemm_df=data.drop(columns=['Run1 (ms)','Run2 (ms)','Run3 (ms)','Run4 (ms)','run_avg'])
sgemm_df.to_csv(r'segmm_product_classification.csv')
sgemm_df.head()
#data info
sgemm_df.info()
#No null values in the data
#checking for NULL values
sgemm_df.isnull().sum() #no NULL values


df_test=sgemm_df.iloc[:,0:14]


#checking variable distribution
for index in range(10):
    df_test.iloc[:,index] = (df_test.iloc[:,index]-df_test.iloc[:,index].mean()) / df_test.iloc[:,index].std();
df_test.hist(figsize= (14,16));
plt.figure(figsize=(14,14))
sns.set(font_scale=1)
sns.heatmap(df_test.corr(),cmap='GnBu_r',annot=True, square = True ,linewidths=.5);
plt.title('Variable Correlation')
#Varibale and predictor
y=np.array(sgemm_df["run_class"])

X=np.array(sgemm_df.iloc[:,0:14])



sc = StandardScaler()
cluster_data = sc.fit_transform(X)

cluster_data[:10]
#kmeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
#kmeans
wcss = []
for i in range(1, 21):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(cluster_data)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 21), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(X)


sgemm_df['clusters1'] = pred_y

sgemm_df.clusters1.unique()
sgemm_df['clusters1'].value_counts()
s = silhouette_score(X, kmeans.labels_)
print('Silhouette Score:', s)

### Run PCA on the data and reduce the dimensions in pca_num_components dimensions
reduced_data = PCA(n_components=2).fit_transform(cluster_data)
results = pd.DataFrame(reduced_data,columns=['pca1','pca2'])

sns.scatterplot(x="pca1", y="pca2", hue=sgemm_df['clusters1'], data=results)
plt.title('K-means Clustering with 2 dimensions')
plt.show()
# Create a PCA instance: pca
pca = PCA(n_components=14)
principalComponents = pca.fit_transform(X)# Plot the explained variances
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_ratio_, color='black')
plt.xlabel('PCA features')
plt.ylabel('variance %')
plt.xticks(features)
print(pca.explained_variance_ratio_)
# Save components to a DataFrame
PCA_components = pd.DataFrame(principalComponents)
plt.scatter(PCA_components[0], PCA_components[1], alpha=.1, color='black')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
ks = range(1, 11)
inertias = []
for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)
    
    # Fit model to samples
    model.fit(PCA_components.iloc[:,:3])
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
plt.plot(ks, inertias, '-o', color='black')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()
PCA_components.iloc[:,:3]
kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(PCA_components.iloc[:,:3])

sgemm_df['clusters_pca'] = pred_y


sgemm_df.clusters_pca.unique()
sgemm_df['clusters_pca'].value_counts()
s = silhouette_score(PCA_components.iloc[:,:3], kmeans.labels_)
print('Silhouette Score:', s)
PCA_components.iloc[:,:3]
import plotly.express as px
#df = px.data.iris()
fig = px.scatter_3d(PCA_components.iloc[:,:3], x=0, y=1, z=2,
            color=pred_y)
fig.show()


rca = GaussianRandomProjection(n_components=4, eps=0.1, random_state=42)
X_rca=rca.fit_transform(X)

plt.scatter(X_rca[0], X_rca[1], alpha=.1, color='black')
plt.xlabel('ICA 1')
plt.ylabel('ICA 2')
## K-Means Clustering Algorithm using RCA
kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(X_rca)

sgemm_df['clusters_ica'] = pred_y        
#print(correct/len(X_ica))
#yp=kmeans.predict(ica_X_train)
plt.scatter(X_rca[:, 0], X_rca[:, 1], c=pred_y, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5)
plt.title('KMeans - ICA')
s_srp = silhouette_score(X, kmeans.labels_)
print('Silhouette Score:', s_srp)
X.shape
y.shape
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
X_train.shape
sel = SelectFromModel(RandomForestClassifier(n_estimators = 100))
sel.fit(X_train, y_train)
boolvec=sel.get_support()
boolvec.astype(bool)
boolvec
input_file=sgemm_df.iloc[:,0:14]
#X_RF=input_file.loc[:, sel.get_support()]
#input_file=sgemm_df.loc[:, sel.get_support()].head()
selected_feat= input_file.columns[(sel.get_support())]
#selected_feat = np.where(boolvec[:,None], X_train,X_train)
len(selected_feat)
print(selected_feat)
#sgemm_df

X_RF=input_file.loc[:, sel.get_support()]

ks = range(1, 11)
inertias = []
for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)
    
    # Fit model to samples
    model.fit(X_RF)
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
plt.plot(ks, inertias, '-o', color='black')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()
kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(X_RF)

sgemm_df['clusters_rf'] = pred_y


sgemm_df.clusters_rf.unique()
sgemm_df['clusters_rf'].value_counts()
s = silhouette_score(X_RF, kmeans.labels_)
print('Silhouette Score:', s)
ICA = FastICA(n_components=2, random_state=42) 
X_ica=ICA.fit_transform(X)
plt.scatter(X_ica[0], X_ica[1], alpha=.1, color='black')
plt.xlabel('ICA 1')
plt.ylabel('ICA 2')
## K-Means Clustering Algorithm using ICA
kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(X_ica)

sgemm_df['clusters_ica'] = pred_y        
#print(correct/len(X_ica))
#yp=kmeans.predict(ica_X_train)
plt.scatter(X_ica[:, 0], X_ica[:, 1], c=pred_y, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5)
plt.title('KMeans - ICA')


s = silhouette_score(X_ica, kmeans.labels_)
print('Silhouette Score:', s)

