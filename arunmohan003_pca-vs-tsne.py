import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('/kaggle/input/iris/Iris.csv')

df = df.drop(columns=['Id'])

print(df.shape)

df.head(3)
#droping the target column

target = df['Species']

data_x = df.iloc[:,:-1]
from sklearn.preprocessing import StandardScaler

scale = StandardScaler()

data = scale.fit_transform(data_x.values)

data.shape
df_matrix = np.asmatrix(data)

print(df_matrix.shape)
covar_matrix = np.cov(df_matrix.T)

print(covar_matrix.shape)
eigvalues, eigvectors = np.linalg.eig(covar_matrix)

print(eigvalues)
print(eigvectors)
top2eig = eigvectors[:,0:2]

top2eig
new_data = data_x.dot(top2eig)

#creating a new dataframe including target

new_df = pd.DataFrame(np.hstack((new_data,np.array(target).reshape(-1,1))),columns=['1st_component','2nd_component','Species'])



new_df.head(2)
#plotting data

sns.scatterplot(new_df['1st_component'],new_df['2nd_component'],hue=new_df['Species'])

plt.title('Scatter-plot')

plt.show()

from sklearn.decomposition import PCA

pca  = PCA(n_components=2)

#here data is scaled data that we did earlier using standard scalar

pca_components = pca.fit_transform(data)

print(pca_components.shape)
#creating a new dataframe including target

new_df_pca = pd.DataFrame(np.hstack((pca_components,np.array(target).reshape(-1,1))),columns=['1st_component','2nd_component','Species'])

new_df_pca.head()

#plotting data

sns.scatterplot(new_df_pca['1st_component'],new_df_pca['2nd_component'],hue=new_df_pca['Species'])

plt.title('Scatter-plot')

plt.show()
from sklearn.manifold import TSNE



tsne = TSNE(n_components=2,perplexity=20,n_iter=1000)

tsne_result = tsne.fit_transform(data)









#creating a new dataframe including target

new_df_tsne = pd.DataFrame(np.hstack((tsne_result,np.array(target).reshape(-1,1))),columns=['1st_component','2nd_component','Species'])

new_df_tsne.head()
#plotting data

sns.scatterplot(new_df_tsne['1st_component'],new_df_tsne['2nd_component'],hue=new_df_tsne['Species'])

plt.title('Scatter-plot')

plt.show()