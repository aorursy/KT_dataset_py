import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
data=pd.read_csv('../input/glass/glass.csv')
data.head(5)

x.shape
y.shape
x = data.iloc[:,0:9]
y = data.iloc[:,9]
sample_data = StandardScaler().fit_transform(x)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

pct = pca.fit_transform(x)

principal_df = pd.DataFrame(pct,columns=['pc1','pc2'])

finaldf= pd.concat([principal_df,data[['Type']]],axis=1)
import seaborn as sn
sn.FacetGrid(finaldf, hue="Type", size=6).map(plt.scatter, 'pc1', 'pc2').add_legend()
plt.show()

pca.n_components = 9
pca_data = pca.fit_transform(sample_data)

percentage_var_explained = pca.explained_variance_ / np.sum(pca.explained_variance_);

cum_var_explained = np.cumsum(percentage_var_explained)

# Plot the PCA spectrum
plt.figure(1, figsize=(6, 4))

plt.clf()
plt.plot(cum_var_explained, linewidth=2)
plt.axis('tight')
plt.grid()
plt.xlabel('n_components')
plt.ylabel('Cumulative_explained_variance')
plt.show()

# TSNE

from sklearn.manifold import TSNE

# Picking the top 1000 points as TSNE takes a lot of time for 15K points
data_1000 = sample_data[0:214,:]
labels_1000 = y[0:214]

model = TSNE(n_components=2, random_state=0,perplexity=30,n_iter=5000)
# configuring the parameteres
# the number of components = 2
# default perplexity = 30
# default learning rate = 200
# default Maximum number of iterations for the optimization = 1000

tsne_data = model.fit_transform(data_1000)


# creating a new data frame which help us in ploting the result data
tsne_data = np.vstack((tsne_data.T, labels_1000)).T
tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "Type"))

# Ploting the result of tsne
sn.FacetGrid(tsne_df, hue="Type", size=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
plt.show()
