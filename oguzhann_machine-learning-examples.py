# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# read csv data

data = pd.read_csv("../input/column_2C_weka.csv")
data.head()
data.info()
data.describe()
sns.countplot(x="class", data=data)

data.loc[:,'class'].value_counts()
color_list = ['yellow' if i=='Abnormal' else 'red' for i in data.loc[:,'class']]

pd.plotting.scatter_matrix(data.loc[:, data.columns != 'class'],

                                       c=color_list,

                                       figsize= [15,15],

                                       diagonal='hist',

                                       alpha=0.5,

                                       s = 200,

                                       marker = '*',

                                       edgecolor= "black")

plt.show()
datax = data[data['class'] =='Abnormal']

x = np.array(datax.loc[:,'pelvic_incidence']).reshape(-1,1)

y = np.array(datax.loc[:,'sacral_slope']).reshape(-1,1)

# Scatter

plt.figure(figsize=[10,10])

plt.scatter(x=x,y=y)

plt.xlabel('pelvic_incidence')

plt.ylabel('sacral_slope')

plt.show()
from sklearn.linear_model import LinearRegression

reg = LinearRegression()

# Predict space

predict_ = np.linspace(min(x), max(x)).reshape(-1,1)

# Fit

reg.fit(x,y)

# Predict

predicted = reg.predict(predict_)

# R^2 

print('R^2 score: ',reg.score(x, y))

# Plot regression line and scatter

plt.plot(predict_, predicted, color='red', linewidth=3)

plt.scatter(x=x,y=y)

plt.xlabel('pelvic_incidence')

plt.ylabel('sacral_slope')

plt.show()
from sklearn.neighbors import KNeighborsClassifier

knnDt = KNeighborsClassifier(n_neighbors = 3)

x,y = data.loc[:,data.columns != 'class'], data.loc[:,'class']

knnDt.fit(x,y)

prediction = knnDt.predict(x)

print('Prediction: {}'.format(prediction))
# train test split

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 1)

knnDt = KNeighborsClassifier(n_neighbors = 3)

x,y = data.loc[:,data.columns != 'class'], data.loc[:,'class']

knnDt.fit(x_train,y_train)

prediction = knnDt.predict(x_test)

#print('Prediction: {}'.format(prediction))

print('With KNN (K=3) accuracy is: ',knnDt.score(x_test,y_test)) # accuracy
neigBr = np.arange(1, 25)

train_accuracy = []

test_accuracy = []

# Loop over different values of k

for i, k in enumerate(neigBr):

    # k from 1 to 25(exclude)

    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit with knn

    knn.fit(x_train,y_train)

    #train accuracy

    train_accuracy.append(knn.score(x_train, y_train))

    # test accuracy

    test_accuracy.append(knn.score(x_test, y_test))



# Plot

plt.figure(figsize=[13,8])

plt.plot(neigBr, test_accuracy, label = 'Testing Accuracy')

plt.plot(neigBr, train_accuracy, label = 'Training Accuracy')

plt.legend()

plt.title('-value VS Accuracy')

plt.xlabel('Number of Neighbors')

plt.ylabel('Accuracy')

plt.xticks(neigBr)

plt.savefig('graph.png')

plt.show()

print("Best accuracy is {} with K = {}".format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))

data2 = pd.read_csv('../input/column_2C_weka.csv')

plt.scatter(data2['pelvic_radius'],data2['degree_spondylolisthesis'])

plt.xlabel('pelvic_radius')

plt.ylabel('degree_spondylolisthesis')

plt.show()
data2_ = data2.loc[:,['degree_spondylolisthesis','pelvic_radius']]

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 2)

kmeans.fit(data2_)

labels = kmeans.predict(data2_)

plt.scatter(data2['pelvic_radius'],data['degree_spondylolisthesis'],c = labels)

plt.xlabel('pelvic_radius')

plt.xlabel('Degree Spondylolisthesis')

plt.show()
dataFrame = pd.DataFrame({'labels':labels,"class":data2['class']})

crossTab = pd.crosstab(dataFrame['labels'],dataFrame['class'])

print(crossTab)
# inertia

inertia_list = np.empty(8)

for i in range(1,8):

    kMeans = KMeans(n_clusters=i)

    kMeans.fit(data2_)

    inertia_list[i] = kMeans.inertia_

plt.plot(range(0,8),inertia_list,'-o')

plt.xlabel('Number of cluster')

plt.ylabel('Inertia')

plt.show()
data_ = pd.read_csv('../input/column_2C_weka.csv')

data3_ = data.drop('class',axis = 1)
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline

scalar = StandardScaler()

kMeans = KMeans(n_clusters = 2)

pipe = make_pipeline(scalar,kMeans)

pipe.fit(data3_)

labels = pipe.predict(data3_)

df = pd.DataFrame({'labels':labels,"class":data_['class']})

ct = pd.crosstab(df['labels'],df['class'])

print(ct)
from scipy.cluster.hierarchy import linkage,dendrogram



merging = linkage(data3_.iloc[200:220,:],method = 'single')

dendrogram(merging, leaf_rotation = 90, leaf_font_size = 6)

plt.show()