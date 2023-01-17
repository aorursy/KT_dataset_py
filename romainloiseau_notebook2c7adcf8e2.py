# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

train[:5]

for col in train.columns[2:]:

    train[col] = (train[col] - min(train[col]))/(max(train[col]) - min(train[col]))
from pandas.plotting import scatter_matrix



scatter_matrix(train.loc[:5000, train.columns[2:10]], figsize = (13, 13), alpha=0.2, diagonal='kde')

plt.show()
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

pca.fit(train[train.columns[2:]])

features= pca.transform(train[train.columns[2:]])

features
for i in range(0, 5000):

    if(train['target'][i] == 0):

        plt.scatter(features[i][0], features[i][1], color = 'r', alpha = 0.5)

    else:

        plt.scatter(features[i][0], features[i][1], color = 'b', alpha = 0.5)

#plt.scatter(np.transpose(features)[0], np.transpose(features)[1])

plt.show()