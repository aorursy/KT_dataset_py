import numpy as np
import pandas as pd
import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt
data = pd.read_csv("../input/customers.csv")
data.head()
data.drop(['Region', 'Channel'], axis = 1, inplace = True)
data.describe()
# Missing data detection
msno.matrix(data,figsize=(10,3))
# Data distribution
sns.boxplot(data=data, orient="v")
# Correlation analasys
corrMatt = data.corr()
mask = np.array(corrMatt)
mask[np.tril_indices_from(mask)] = False
fig,ax= plt.subplots()
fig.set_size_inches(20,10)
sns.heatmap(corrMatt, mask=mask,vmax=.8, square=True,annot=True)
# Scatterplot
mx_plot = sns.pairplot(data, diag_kind="kde", height=1.6)
mx_plot.set(xticklabels=[])
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from sklearn.decomposition import PCA 
X = data.drop(["Grocery"], axis = 1)
X.head()
# Scaling data
scaler = StandardScaler()
X = scaler.fit_transform(X)
# Create dendragram
dendagram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title("Dendagram")
plt.show()
# Creating model
model = AgglomerativeClustering(n_clusters = 7, affinity = 'euclidean', linkage = 'ward')
y = model.fit_predict(X)

pca_2 = PCA(2) 
plot_columns = pca_2.fit_transform(data)
plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=model.labels_)
plt.xlabel('Canonical variable 1')
plt.ylabel('Canonical variable 2')
plt.title('Scatterplot of Canonical Variables for 5 Clusters')
plt.show()
