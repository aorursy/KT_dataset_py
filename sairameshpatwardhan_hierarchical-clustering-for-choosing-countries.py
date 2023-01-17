import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
#%matplotlib inline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering

country_data = pd.read_csv('../input/unsupervised-learning-on-country-data/Country-data.csv')
print(country_data.shape)
country_data.head()
country_data.isnull().sum()
country_data.info()   
country_data.describe(percentiles=[.25,.5,.75,.90,.95,.99])
plt.figure(figsize=(15,8))
sns.heatmap(country_data.corr(),annot=True,cmap='Blues')
plt.show()
data = country_data.drop('country', axis=1)
sc=StandardScaler()
data[data.columns] = sc.fit_transform(data[data.columns])

Pos_features= data[['exports','health','imports','income','gdpp','life_expec']]
Neg_features= data[['child_mort','inflation','total_fer']]
  
pca = PCA(n_components = 1) 
  
PC1 = pca.fit_transform(Pos_features) 
PC2 = pca.fit_transform(Neg_features) 

plt.plot(PC1,PC2,'ro')
plt.xlabel('PC1') 
plt.ylabel('PC2') 
# Creating new datframe combining PC1 and PC2
data1= pd.DataFrame(PC1)
data2= pd.DataFrame(PC2)
pca_data=pd.concat([data1, data2], axis=1)
pca_data.columns= ['PC1','PC2']
plt.figure(figsize=(50, 12))  
plt.title("Dendrogram")  
dend = shc.dendrogram(shc.linkage(pca_data, method='ward'))
hcluster = AgglomerativeClustering(n_clusters=6, affinity='euclidean', linkage='ward')  
hcluster.fit_predict(pca_data)
Y_hc = hcluster.labels_
pca_data['Pred_hc']= Y_hc

sns.scatterplot(x='PC1',y='PC2',hue='Pred_hc',legend='full',data=pca_data,palette="muted")
country_data['Pred_hc']= Y_hc
country_data.to_csv('result.csv') 

