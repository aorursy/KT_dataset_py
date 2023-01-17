# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_train=pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

df_test=pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

submission=pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')
df_train.info()
df_train.head()
data=df_train.drop('label',axis=1)

label=df_train['label']
plt.figure(figsize=(7,7))

idx = 25



grid_data = data.iloc[idx].as_matrix().reshape(28,28)  # reshape from 1d to 2d pixel array

plt.imshow(grid_data, interpolation = "none", cmap = "gray")

plt.show()





print("label "+str(label[idx]))
from sklearn.preprocessing import StandardScaler

standardized_data = StandardScaler().fit_transform(data)

print(standardized_data.shape)
from sklearn import decomposition

pca = decomposition.PCA()

# configuring the parameteres

# the number of components = 2

pca.n_components = 2

pca_data = pca.fit_transform(data)



# pca_reduced will contain the 2-d projects of simple data

print("shape of pca_reduced.shape = ", pca_data.shape)
pca_data = np.vstack((pca_data.T, label)).T
import seaborn as sn

pca_df = pd.DataFrame(data=pca_data, columns=("1st_component", "2nd_component", "label"))

sn.FacetGrid(pca_df, hue="label", size=6).map(plt.scatter, '1st_component', '2nd_component').add_legend()

plt.show()
from sklearn.manifold import TSNE

model = TSNE(n_components=2, random_state=0)

tsne_data = model.fit_transform(data)

tsne_data = np.vstack((tsne_data.T, label)).T

tsne_df = pd.DataFrame(data=tsne_data, columns=("Dimension_1", "Dimension_2", "label"))
sn.FacetGrid(tsne_df, hue="label", size=6).map(plt.scatter, 'Dimension_1', 'Dimension_2').add_legend()

plt.show()