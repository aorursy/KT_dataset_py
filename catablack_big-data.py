# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

from sklearn.cluster import KMeans

import numpy as np

import seaborn as sns

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

from sklearn.preprocessing import StandardScaler 

from sklearn import svm



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print("------READING DATA-------")



data = pd.read_csv("../input/master.csv")

#data = data.drop(columns="HDI for year")

data = data.drop(columns="country")

data = data.drop(columns=" gdp_for_year ($) ")

data = data.drop(columns="age")

data = data.drop(columns="year")

data = data.drop(columns="sex")

data = data.drop(columns="generation")

data = data.drop(columns="gdp_per_capita ($)")

data = data.drop(columns="country-year")



data.describe()

gender = {'male': 1,'female': 2} 

generation = {'Boomers':1, 'G.I. Generation':2, 'Generation X':3, "Generation Z":4, 'Millenials':5, 'Silent':6}

data.rename(columns={"HDI for year": 'HDI'}, inplace= True)

# traversing through dataframe 

# Gender column and writing 

# values where key matches 

#data.sex = [gender[item] for item in data.sex]

#data.generation = [generation[item] for item in data.generation] 

data = data[data['HDI'].notnull()]

#print(data) 

# Any results you write to the current directory are saved as output.
print("Dimension of dataset: data.shape")

data.dtypes
data1 = data[['suicides_no','suicides/100k pop','population','HDI']]

cor = data.corr()

sns.heatmap(cor, square = False)
dataHotEnc = data
#K means Clustering 

def doKmeans(X, nclust=2):

    model = KMeans(nclust)

    model.fit(X)

    clust_labels = model.predict(X)

    cent = model.cluster_centers_

    return (clust_labels, cent)



clust_labels, cent = doKmeans(dataHotEnc, 4)

kmeans = pd.DataFrame(clust_labels)
dataHotEnc = dataHotEnc.set_index(kmeans.index)

dataHotEnc.insert((dataHotEnc.shape[1]),'kmeans',kmeans)

#dataHotEnc
fig = plt.figure()

ax = fig.add_subplot(111)

scatter = ax.scatter(dataHotEnc['suicides/100k pop'],dataHotEnc['population'],

                     c=kmeans[0],s=50)

ax.set_title('K-Means Clustering')

ax.set_ylabel('population')

ax.set_xlabel('suicides/100k pop')

plt.colorbar(scatter)