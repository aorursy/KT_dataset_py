import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn import manifold

from sklearn.preprocessing import StandardScaler

from sklearn import decomposition
data = pd.read_csv('../input/digit-recognizer/train.csv')

data.head()
df_labels = data.label

df_data = data.drop('label', axis = 1)

#Count plot for the labels 

sns.countplot(df_labels)
#extracting top 10000 data points 

df_data = df_data.head(10000)

df_labels = df_labels.head(10000)

pixel_df = StandardScaler().fit_transform(df_data)

pixel_df.shape
sample_data = pixel_df
pca = decomposition.PCA(n_components = 2, random_state = 42)
pca_data = pca.fit_transform(sample_data)

print("shape of pca_reduced.shape = ", pca_data.shape)
# attaching the label for each 2-d data point 

pca_data = np.column_stack((pca_data, df_labels))



# creating a new data frame for plotting of data points

pca_df = pd.DataFrame(data=pca_data, columns=("X", "Y", "labels"))

print(pca_df.head(10))

sns.FacetGrid(pca_df, hue="labels", size=6).map(plt.scatter, 'X', 'Y').add_legend()

plt.show()
tsne = manifold.TSNE(n_components = 2, random_state = 42, verbose = 2, n_iter = 2000)

transformed_data = tsne.fit_transform(sample_data)
#Creation of new dataframe for plotting of data points

tsne_df = pd.DataFrame(

    np.column_stack((transformed_data, df_labels)),

    columns = ['x', 'y', 'labels'])

tsne_df.loc[:, 'labels']= tsne_df.labels.astype(int)

print(tsne_df.head(10))

grid = sns.FacetGrid(tsne_df, hue='labels', size = 8)

grid.map(plt.scatter, 'x', 'y').add_legend()