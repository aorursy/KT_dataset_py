# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import matplotlib.pyplot as plt

import scipy.linalg as la

import seaborn as sns

from sklearn.decomposition import PCA



import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import statistics

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        

        url = os.path.join(dirname, filename)

        print(url)



import pandas as pd



# load dataset into Pandas DataFrame

# train_data = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target'])

t_data = pd.read_csv(url)
train_data= t_data.loc[:,:].values

ids = train_data[:,0]

train_data = train_data[:,1:5]

plt.scatter(ids, train_data[:,0])

plt.scatter(ids, train_data[:,1])

plt.scatter(ids, train_data[:,2])

plt.scatter(ids, train_data[:,3])
# plot_data["Pre"] = y_data

# sns.pairplot(t_data, hue='Species', palette='OrRd')

sns.pairplot(t_data, hue='Species', palette='OrRd')
type(train_data[1,2]) #Comes out to be float

train_mean = np.zeros((1,train_data.shape[1]))



train_mean[0,0] = statistics.mean(train_data[:,0])

train_mean[0,1] = statistics.mean(train_data[:,1])

train_mean[0,2] = statistics.mean(train_data[:,2])

train_mean[0,3] = statistics.mean(train_data[:,3])

train_mean.shape
# New Data

new_train_data = train_data-train_mean

new_train_data.shape

new_train_data_t = new_train_data.T #For multiplication purpose
mean_data = pd.DataFrame(new_train_data)

sns.pairplot(mean_data, palette='OrRd')
train_cov = np.cov(new_train_data.astype(float).T)

train_cov
eg, ev = la.eig(train_cov)

ev=ev.T

eg, ev
eg_list = eg.tolist()

mop = eg_list.index(max(eg))

mop

# max_eigvector = np.zeros((4,1))

max_eigvector = ev[mop]

# max_eigvector = max_eigvector

type(max_eigvector)

# Conversion into matrix

egv = np.asmatrix(max_eigvector)

# egv = egv

plt.plot(ev)
plt.plot(max_eigvector)
finale = (egv)*(new_train_data_t)

finale = finale.T

# finale = max_eigvector.dot(new_train_data)

finale.shape

# new_train_data.shape

# finale

# new_train_data.T.shape

print('Before applying PCA, shape was: ', train_data.shape)

print('After applying PCA, shape of the data is: ', finale.shape)
finale[:,0].shape
print('Total data preserved = ', max(eg)/np.sum(eg)*100)
# Dataframes and plots

# in_data = t_data.loc[:,:].values

in_data_manual = pd.DataFrame(t_data)

new_data_manual = pd.DataFrame(new_train_data)

final_data_manual = pd.DataFrame(finale)



features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

target = 'Species'

x_data = t_data[features]

y_data = t_data[target]



final_data_manual['Species'] = y_data
# Plottng manual pca graph

sns.pairplot(final_data_manual, hue='Species', palette='OrRd')


pca_data = x_data

output = y_data

model = PCA(n_components=3)

pca_fit = model.fit_transform(x_data)

pca_df = pd.DataFrame(data = pca_fit

             , columns = ['pc1', 'pc2', 'pc3'])



pca_df.head(5)

pca_df["Pre"] = output

sns.pairplot(pca_df, hue='Pre', palette='OrRd')
g = sns.FacetGrid(pca_df, hue='Pre', hue_kws={"marker": ["^", "v", "*"]})

# g = sns.FacetGrid(plot_data, col="Prediction", row="Date")

g.map(plt.scatter, "pc1", "pc2", alpha=.7)

g.add_legend();
g = sns.FacetGrid(pca_df, hue='Pre', hue_kws={"marker": ["^", "v", "*"]})

g.map(plt.scatter, "pc1", "pc3", alpha=.7)

g.add_legend();
g = sns.FacetGrid(pca_df, hue='Pre', hue_kws={"marker": ["^", "v", "*"]})

g.map(plt.scatter, "pc2", "pc3", alpha=.7)

g.add_legend();
eg_pca, ev_pca = la.eig(model.get_covariance())

ev_pca = ev_pca.T

print(eg_pca, ev_pca)

print('Total data preserved with built in PCA= ', max(eg_pca)/np.sum(eg_pca)*100)