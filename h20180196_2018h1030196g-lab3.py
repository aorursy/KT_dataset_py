import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.metrics import auc, roc_curve, roc_auc_score

df=pd.read_csv("/kaggle/input/eval-lab-3-f464/train.csv")


y=df["Satisfied"]

df.drop(["Satisfied"],inplace=True,axis=1)
df["TotalCharges"]=pd.to_numeric(df["TotalCharges"],errors='coerce')

df["TotalCharges"]=df["TotalCharges"].fillna(0.0)

df=pd.get_dummies(df)
df.head()

df["Satisfied"]=y
# df.info()
plt.figure(figsize=(15, 15))

sns.heatmap(df.corr(), cmap=sns.diverging_palette(200,10,as_cmap=True))
X=df.drop(['custId','gender_Female','gender_Male','Internet_No','Internet_Yes','HighSpeed_No', 'HighSpeed_Yes', 'HighSpeed_No internet', 'Satisfied'], axis=1)

y=df['Satisfied']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# X.info()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_s = scaler.fit_transform(X)
################################################################################
# from sklearn.cluster import KMeans



# kmeans = KMeans(n_clusters=2)

# kmeans.fit(X)
# km_pred=kmeans.predict(X)
# roc_auc_score(pred_y, y)
# from sklearn.decomposition import PCA

# import pylab as pl

# pca = PCA(n_components=2).fit(df)

# pca_2d = pca.transform(df)



# for i in range(0, pca_2d.shape[0]):

#     if df.Satisfied[i] == 1:

#         c1 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='r',

#         marker='+')

#     elif df.Satisfied[i] == 0:

#         c2 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='g',

#         marker='o')

# pl.legend([c1, c2],['Cluster 1', 'Cluster 0'])

# pl.title('K-means clusters')

# pl.show()
# from sklearn.decomposition import PCA

# import pylab as pl

# pca = PCA(n_components=2).fit(X)

# pca_2d = pca.transform(X)



# for i in range(0, pca_2d.shape[0]):

#     if kmeans.labels_[i] == 1:

#         c1 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='r',

#         marker='+')

#     elif kmeans.labels_[i] == 0:

#         c2 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='g',

#         marker='o')

#     elif kmeans.labels_[i] == 2:

#         c3 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='b',

#         marker='*')

# pl.legend([c1, c2],['Cluster 1', 'Cluster 0', 'cluster3'])

# pl.title('K-means clusters')

# pl.show()
df2=pd.read_csv("/kaggle/input/eval-lab-3-f464/test.csv")
df2["TotalCharges"]=pd.to_numeric(df2["TotalCharges"],errors='coerce')

df2["TotalCharges"]=df2["TotalCharges"].fillna(0.0)

df2=pd.get_dummies(df2)

df3=df2.drop(['custId','gender_Female','gender_Male','Internet_No','Internet_Yes','HighSpeed_No', 'HighSpeed_Yes', 'HighSpeed_No internet'], axis=1)
X_new=pd.concat([X, df3])
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_s = scaler.fit_transform(X_new)
##############################################################################
# from sklearn.cluster import KMeans



# kmeans = KMeans(algorithm='auto',n_clusters=2, max_iter=1200, random_state=111)

# kmeans.fit(X_s)
# km_pred=kmeans.predict(X_s)
# comp=km_pred[0:4930]

# ans=km_pred[4930:]
# roc_auc_score(comp,y)
###############################################################################
# from sklearn.cluster import AgglomerativeClustering
# clustering = AgglomerativeClustering(affinity='manhattan', linkage='single')

# clustering.fit(X_s)
# ag_pred=clustering.labels_
# comp=ag_pred[:4930]

# ans=ag_pred[4930:]
# roc_auc_score(comp, y)
############################################################################
# from sklearn.mixture import GaussianMixture
# gmm=GaussianMixture(max_iter=1000)

# gmm.fit(X)
# gmm_pred=gmm.predict(X)
# comp=gmm_pred[0:4930]

# ans=gmm_pred[4930:]
# comp.sum()
# roc_auc_score(comp, y)
########################################################################################
# from sklearn.cluster import Birch 
# birch= Birch(n_clusters=2, threshold=1.10, branching_factor=6023)

# birch.fit(X_s)

# b_pred=birch.labels_

# comp=b_pred[0:4930]

# ans=b_pred[4930:]

# roc_auc_score(comp, y)
#################################################################################
from sklearn.cluster import SpectralClustering

model = SpectralClustering(n_clusters=2,  affinity='poly',

                           assign_labels='kmeans',n_init=100, gamma=0.5,n_neighbors=35, n_jobs=-1, random_state=42, eigen_solver='arpack')

sc_pred = model.fit_predict(X_s)
comp=sc_pred[0:4930]

ans=sc_pred[4930:]

roc_auc_score(comp, y)
####################################################################################
# from sklearn.cluster import MiniBatchKMeans



# minikmeans = MiniBatchKMeans(n_clusters=3,batch_size=1000,max_iter=102, max_no_improvement=100)

# minikmeans.fit(X)
# mini_pred=minikmeans.predict(X)

# comp=mini_pred[0:4930]

# ans=mini_pred[4930:]

# roc_auc_score(comp, y)

# # fpr,tpr, thresholds = roc_curve(comp,y)

# # auc(fpr,tpr)
np.unique(comp)
df2['custId'].shape
ans.shape
sol=pd.DataFrame()

sol['custId']=df2['custId']

sol['Satisfied']=ans
sol.to_csv("sol.csv", index=False)