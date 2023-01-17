# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
train_set = pd.read_csv('/kaggle/input/digit-recognizer/train.csv', sep = ',', header = 0)

t_sne_set = train_set.sample(10000)

labels = pd.DataFrame(train_set['label'].copy())

train_set.drop(['label'], inplace = True, axis = 1)

train_set.head()
labels.head()
train_set.iloc[:, :]/=255.0
sns.countplot(labels['label'])
pca = PCA(n_components = 2)

reduced_data = pca.fit_transform(train_set)
pca_data = np.concatenate([reduced_data, labels.values], axis = 1)

pca_df = pd.DataFrame(data = pca_data, columns = ('x', 'y', 'label'))
sns.FacetGrid(pca_df, hue="label", size=15).map(plt.scatter, 'x', 'y', alpha = 0.7).add_legend()
plt.show()
t_sne_labels = pd.DataFrame(t_sne_set['label'].copy())

t_sne_set.drop(['label'], inplace = True, axis = 1)
t_sne_set.iloc[:, :]/=255.0

tsne = TSNE(n_components = 2, verbose = 2, n_iter = 2000)
transformed_data = tsne.fit_transform(t_sne_set)
t_sne_df = pd.DataFrame(data = np.concatenate([transformed_data, t_sne_labels.values], axis = 1), columns = ['x', 'y', 'label']) 

sns.FacetGrid(t_sne_df, hue="label", size=15).map(plt.scatter, 'x', 'y', alpha = 0.7).add_legend()

plt.show()
transformed_data = UMAP(n_neighbors=5, min_dist=0.3, metric='correlation').fit_transform(train_set)

umap_df = pd.DataFrame(data = np.concatenate([transformed_data, labels.values], axis = 1), columns = ['x', 'y', 'label'])

sns.FacetGrid(umap_df, hue="label", size=15).map(plt.scatter, 'x', 'y', alpha = 0.7).add_legend()

plt.show()
import tensorflow as tf

encoder = tf.keras.models.Sequential()

encoder.add(tf.keras.layers.Dense(units = 64, activation = 'relu', kernel_initializer = 'glorot_uniform'))
encoder.add(tf.keras.layers.Dense(units = 32, activation = 'relu', kernel_initializer = 'glorot_uniform'))
encoder.add(tf.keras.layers.Dense(units = 8, activation = 'relu', kernel_initializer = 'glorot_uniform'))

latent_layer = tf.keras.models.Sequential(tf.keras.layers.Dense(units = 2, activation = 'linear', kernel_initializer = 'glorot_uniform'))

decoder = tf.keras.models.Sequential()

decoder.add(tf.keras.layers.Dense(units = 8, activation = 'relu', kernel_initializer = 'glorot_uniform'))
decoder.add(tf.keras.layers.Dense(units = 32, activation = 'relu', kernel_initializer = 'glorot_uniform'))
decoder.add(tf.keras.layers.Dense(units = 64, activation = 'relu', kernel_initializer = 'glorot_uniform'))
decoder.add(tf.keras.layers.Dense(units = 784, activation = 'sigmoid', kernel_initializer = 'glorot_uniform'))

auto_encoder = tf.keras.models.Sequential([encoder, latent_layer, decoder])
auto_encoder.build(input_shape = (None, 784))

auto_encoder.summary()
auto_encoder.compile(loss = 'binary_crossentropy', optimizer = 'adam')

auto_encoder.fit(x = train_set.values, y = train_set.values, batch_size = 64, epochs = 25)
transformed_data = tf.keras.models.Sequential([encoder, latent_layer]).predict(x = train_set.values)

ae_df = pd.DataFrame(data = np.concatenate([transformed_data, labels.values], axis = 1), columns = ['x', 'y', 'label'])

sns.FacetGrid(ae_df, hue="label", size=15).map(plt.scatter, 'x', 'y', alpha = 0.7).add_legend()

plt.show()
