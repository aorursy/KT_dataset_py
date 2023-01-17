import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from scipy.linalg import eigh

import random

from sklearn.manifold import TSNE
df = pd.read_csv('../input/train.csv')

df.info()
print(df.head())

df = df[:1000]
"Seperating labels and feature vectors"

label = df.label

data = df.drop('label', axis = 1)
def print_index(id):

    idx = id

    grid_data = data.iloc[idx].values.reshape(28,28)

    plt.imshow(grid_data, interpolation = None, cmap = 'gray')

    plt.title(label[idx])
print_index(random.randint(0, 1000)) 
standardized_data = StandardScaler().fit_transform(data)

print(standardized_data.shape)
cov_matrix = np.dot(standardized_data.T, standardized_data)

values, vectors = eigh(cov_matrix, eigvals = (782, 783))

new_data = np.dot(vectors.T, standardized_data.T)

print(new_data.shape)

xdash = np.vstack((new_data, label)).T

print(xdash.shape)
df = pd.DataFrame(data = xdash, columns = ('1st Principal', '2nd Principal', 'Labels'))

sns.FacetGrid(df, hue = 'Labels', height = 7).map(plt.scatter, '1st Principal', '2nd Principal').add_legend()

plt.title('PCA on MNIST')

plt.show()
model = TSNE(n_components = 2, random_state = 0)

tsne_model = model.fit_transform(standardized_data)
tsne_data = np.vstack((tsne_model.T, label)).T

# tsne_data.shape

# labels.shape

tse_df = pd.DataFrame(data= tsne_data, columns=('Dim_1', 'Dim_2', 'Labels'))
sns.FacetGrid(tse_df, hue = 'Labels', height = 6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()

plt.show()