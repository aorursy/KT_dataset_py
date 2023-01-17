# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
mnist_df = pd.read_csv('../input/train.csv')
mnist_df.head()
mnist_labels = mnist_df['label']
mnist_pixels = mnist_df.drop('label', axis = 1)

# standardize mnist dataset

from sklearn.preprocessing import StandardScaler
mnist_pixels_std_df = StandardScaler().fit_transform(mnist_pixels)
print(mnist_pixels_std_df.shape)

# Clip 1000 datapoints

mnist_pixels_tsne = mnist_pixels_std_df#mnist_pixels_std_df[0:1000,:]
mnist_labels_tsne = mnist_labels#mnist_labels[:1000]
print("MNIST labels : ", mnist_labels_tsne.shape)
print("MNIST features : ", mnist_pixels_tsne.shape)

# t-SNE
from sklearn.manifold import TSNE
import timeit
# code you want to evaluate
# D-dash = 2
# default perplexity = 30
# default learning rate = 200
# default no. of iteration for optimization = 1000

start_time = timeit.default_timer()

model = TSNE(n_components = 2, random_state = 0)
tsne_model = model.fit_transform(mnist_pixels_tsne).T

elapsed = timeit.default_timer() - start_time
# visualizing t-SNE
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

tsne_data = np.vstack((tsne_model, mnist_labels_tsne)).T
tsne_df = pd.DataFrame(data = tsne_data, columns=('Dimention 1', 'Dimention 2', 'Label'))
sns.FacetGrid(tsne_df, size = 8, hue = 'Label').map(plt.scatter, 'Dimention 1', 'Dimention 2').add_legend()

print('\n\nElapsed time for t-SNE visualization :', elapsed, 'seconds')