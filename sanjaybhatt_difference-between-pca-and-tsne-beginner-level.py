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
data=pd.read_csv('../input/mnist-in-csv/mnist_train.csv')
data.head()
data.shape
label=data['label']
data=data.drop(['label'], axis=1) #axis 1 means doing operaion in columns
x=data.head(15000)

y=label.head(15000)
from sklearn.preprocessing import StandardScaler

std_x=StandardScaler().fit_transform(x)

std_x.shape
type(std_x)
from sklearn import decomposition

pca=decomposition.PCA()

pca.n_components=2

pca_x=pca.fit_transform(std_x)
pca_data=np.vstack((pca_x.T, y))

pca_df=pd.DataFrame(pca_data.T, columns=['first', 'second', 'label'])
pca_df.head()
import matplotlib.pyplot as plt

import seaborn as sns
ax=sns.FacetGrid(pca_df, hue='label', height=6).map(plt.scatter, 'first', 'second').add_legend()

plt.show()
from sklearn.manifold import TSNE

tsne=TSNE(n_components=2, random_state=0)

tsne_x=tsne.fit_transform(std_x)

tsne_data=np.vstack((tsne_x.T, y))

tsne_df=pd.DataFrame(tsne_data.T, columns=['t_first', 't_second', 't_label'])
tsne_df.head()
tsne_df.shape
ax=sns.FacetGrid(tsne_df, hue='t_label', height=6).map(plt.scatter, 't_first', 't_second').add_legend()

plt.show()