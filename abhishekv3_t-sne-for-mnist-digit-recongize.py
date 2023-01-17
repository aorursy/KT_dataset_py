# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/digit-recognizer/train.csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
df.head(10)

# save the labels into l
l = df['label']
# store the pixel into d, except l
d = df.drop("label", axis=1)
# d
# check shape
print(l.shape)
print(d.shape)
import matplotlib.pyplot as plt

# display number
plt.figure(figsize=(7,7))
idx=1

# reshape pixel from 1d to 2d pixel array
grid = d.iloc[idx].to_numpy().reshape(28, 28)
plt.imshow(grid, interpolation="none", cmap="gray")
plt.show()

print(l[idx])
labels = l.head(1000)
data = d.head(1000)

print("shape of sample data =" , data.shape)
# Standardizing the data
from sklearn.preprocessing import StandardScaler
standardized_data = StandardScaler().fit_transform(data)
print(standardized_data.shape)
from sklearn.manifold import TSNE

# pickle top 1k points as tSNE
data_1k = standardized_data[0:1000, :]
labels_1k = labels[0:1000]


model = TSNE(n_components=2, random_state=0)

tsne_data = model.fit_transform(data_1k)
tsne_data = np.vstack((tsne_data.T, labels_1k)).T
tsne_df = pd.DataFrame(data=tsne_data, columns = ("Dim_1", "Dim_2", "label"))


import seaborn as sn
import matplotlib.pyplot as plt


sn.FacetGrid(tsne_df, hue="label", height=6).map(plt.scatter, "Dim_1", "Dim_2").add_legend()
plt.show()
model = TSNE(n_components=2, random_state=0, perplexity=24, n_iter=580)
tsne_data = model.fit_transform(data_1k) 

# creating a new data fram which help us in ploting the result data
tsne_data = np.vstack((tsne_data.T, labels_1k)).T
tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "label"))

# Ploting the result of tsne
sn.FacetGrid(tsne_df, hue="label", height=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
plt.title('With perplexity = 50')
plt.show()
