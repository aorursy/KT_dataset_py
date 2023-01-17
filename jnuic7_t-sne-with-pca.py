# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.manifold import TSNE

from sklearn.decomposition import PCA

from sklearn.neighbors import NearestNeighbors

import matplotlib.pyplot as plt

import seaborn as sns

from keras.models import Sequential

from keras.layers import Dense,LSTM

from sklearn.preprocessing import MinMaxScaler

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#First going to process the data into image matricies

data = pd.read_csv('../input/fashion-mnist_train.csv')

X_train = data.iloc[:,1:].values

y_train = data.label.values

print(X_train.shape[0])

#Reshape X, 28 px by 28 px, rescale to 0-1(same for digit mnist)

X_train = X_train.reshape(X_train.shape[0],28,28)

X_train = X_train/255
#Since the sample is too large(will crash my box), lets reduce the sample size

pca = PCA(n_components=30)

Xp = pca.fit_transform(data.iloc[:,1:])

print(Xp.shape)
n_sne = 1000

for i in range(5):

    tsne = TSNE(n_components=2,verbose=1,perplexity=10*i,n_iter=1000)

    X_tsne = tsne.fit_transform(Xp[:n_sne])

    Xf = pd.DataFrame(X_tsne)

    Xf.columns = ["comp1","comp2"]

    Xf['labels'] = y_train[:n_sne]

    sns.lmplot("comp1","comp2",hue="labels",data=Xf,fit_reg=False)
n_tsne = 1000

#Since 60000 is to large to feed through TSNE, will limit it to 1,000 reference greatly to https://www.kaggle.com/dualphase/t-sne

tsne = TSNE(n_components=2,verbose=1,perplexity=40,n_iter=1000)

Xt = tsne.fit_transform(Xp[:n_tsne])

print(Xt.shape)
#Create a data frame

Xdf = pd.DataFrame(Xt)

Xdf.columns = ['comp1','comp2']

Xdf['labels'] = y_train[:n_tsne]

sns.lmplot('comp1','comp2',hue='labels',data=Xdf,fit_reg=False)

#Lets try nearest neighbors with the set, pick random index

knn = NearestNeighbors(500)

knn.fit(Xt[:500])

imgt = Xt[5].reshape(1,Xt[4].shape[0])

dist,neigh = knn.kneighbors(imgt.reshape(1,-1))

neigh = neigh[0]





#Does a pretty good job of clustering long sleeve shirts, ideally would like to use a bigger sample of the 60000

for i,j in enumerate(neigh[:36]):

    plt.subplot(6,6,i+1)

    plt.imshow(X_train[j],cmap='gray')

