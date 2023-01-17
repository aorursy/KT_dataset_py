# Partialy reused code from https://www.kaggle.com/kanncaa1/machine-learning-tutorial-for-beginners/notebook

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/column_2C_weka.csv')

plt.style.use('ggplot')
#https://pandas.pydata.org/pandas-docs/stable/reference/frame.html

data.count() # Spondylolisthesis -> kręgozmyk
print(data['class'].unique())

print(data['class'].value_counts())

data['class'].value_counts().plot.bar()

data.head(10)
data.info()
data.describe()
data.plot.scatter(x='pelvic_radius', y='sacral_slope')

data.plot(x='degree_spondylolisthesis', y='lumbar_lordosis_angle', kind='scatter')

color_list = ['red' if i=='Abnormal' else 'green' for i in data.loc[:,'class']]

pd.plotting.scatter_matrix(data.loc[:, data.columns != 'class'],

                                       c=color_list,

                                       figsize= [15,15],

                                       diagonal='hist',

                                       alpha=0.5,

                                       s = 200,

                                       marker = '*',

                                       edgecolor= "black")

plt.show()
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)

atrybuty,etykiety = data.loc[:,data.columns != 'class'], data.loc[:,'class'] 

knn.fit(atrybuty,etykiety)

prediction = knn.predict(atrybuty)

print('Prediction: {}'.format(prediction))

knn.score(atrybuty, etykiety) # czemu wynik nie jest równy 1?
# Do wypelnienia
atrybuty
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 1) # random_state musi byc 1, zeby moc odtworzyc ten sam podział po restarcie kernela
knn_trening = KNeighborsClassifier(n_neighbors = 3)

knn.fit(x_train,y_train)

prediction = knn.predict(x_test)

print('With KNN (K=3) accuracy is: ',knn.score(x_test,y_test))