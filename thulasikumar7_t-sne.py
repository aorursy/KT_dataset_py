# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

d0=pd.read_csv("../input/mnist_train.csv")

# Any results you write to the current directory are saved as output.
data=d0.drop("label",axis=1)
labels=d0["label"]
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
std_data=StandardScaler().fit_transform(data)
from sklearn.manifold import TSNE




data_1000=std_data[0:1000,:]
data_1000.shape
labels_1000=labels[0:1000]
labels.shape
model=TSNE(n_components=2,random_state=0)

tsne_data=model.fit_transform(data_1000)
tsne_data.shape
tsne_data=np.vstack((tsne_data.T,labels_1000)).T
tsne_data.shape
tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "label"))
from sklearn.manifold import TSNE

# Picking the top 1000 points as TSNE takes a lot of time for 15K points
data_1000 = standardized_data[0:1000,:]
labels_1000 = labels[0:1000]

model = TSNE(n_components=2, random_state=0)
# configuring the parameteres
# the number of components = 2
# default perplexity = 30
# default learning rate = 200
# default Maximum number of iterations for the optimization = 1000

tsne_data = model.fit_transform(data_1000)


# creating a new data frame which help us in ploting the result data
tsne_data = np.vstack((tsne_data.T, labels_1000)).T
tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "label"))

# Ploting the result of tsne
sn.FacetGrid(tsne_df, hue="label", size=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
plt.show()