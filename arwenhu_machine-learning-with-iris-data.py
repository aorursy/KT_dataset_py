# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib as mpl

from matplotlib.widgets import Slider, Button, RadioButtons

import seaborn as sns

from sklearn.decomposition import PCA



%matplotlib inline

plt.rcParams['figure.figsize'] = (10.0, 6.0)  # set default size of plots

#plt.rcParams['image.aspect'] = 1.3

plt.rcParams['image.interpolation'] = 'nearest' # 设置 interpolation style

#plt.rcParams['image.cmap'] = 'gray'

plt.rcParams['savefig.dpi'] = 300  #图片像素

plt.rcParams['figure.dpi'] = 300

plt.rcParams['font.size'] = 20

plt.rcParams['font.weight'] = 'normal'

plt.rcParams['lines.markersize'] = 8

plt.rcParams['axes.spines.top'] = False

plt.rcParams['axes.spines.right'] = False #display only a left and bottom box border

plt.rcParams['legend.edgecolor'] = 'white'

plt.rcParams['axes.axisbelow'] = False

plt.rcParams['grid.color'] = 'white'



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
iris = pd.read_csv('../input/Iris.csv', header = 0)

print(iris.info())

print(iris.describe())

print(iris.head())
sns.relplot(

    x="SepalLengthCm",

    y="SepalWidthCm",

    hue="Species",

    dashes=False,

    markers=True,

    kind="scatter",

    data=iris,

    height=5,

    aspect=1.3)
sns.relplot(

    x="PetalLengthCm",

    y="PetalWidthCm",

    hue="Species",

    dashes=False,

    markers=True,

    kind="scatter",

    data=iris,

    height=5,

    aspect=1.3)
iris.drop('Id', axis=1).plot.hist(edgecolor='white', linewidth=1.2, alpha=0.5, bins=50)
iris.drop('Id', axis=1).hist(edgecolor='white', linewidth=1.2, grid=False, alpha=0.5)
iris_melt = iris.drop('Id', axis=1).melt(

    id_vars='Species',

    value_vars=iris.drop(['Id', 'Species'], axis=1).columns,

    value_name="value")

sns.violinplot(

    data=iris_melt, x='variable', y='value', hue='Species')

plt.xticks(rotation=45)
sns.heatmap(iris.drop('Id', axis=1).corr(), annot=True, center=0, vmin=-1, vmax=1, cmap='RdBu_r')
pca = PCA(n_components=2) #PCA(n_components='mle') #

pca.fit(iris.drop("Species",axis=1))

print(pca.explained_variance_ratio_) 
iris_r = pca.transform(iris.drop("Species",axis=1))

iris_r = pd.DataFrame(iris_r,columns=["PC1","PC2"])

f = lambda x: round(x,1)

iris_r = iris_r.applymap(f)

print(iris_r.head())

iris_r = pd.concat([iris_r,iris["Species"]],axis=1)

plt.figure(figsize = (10,6))

sns.stripplot(x="PC1",y="PC2",hue="Species",data=iris_r)

plt.xticks(rotation=45)
from sklearn import metrics

from sklearn.linear_model import LogisticRegression  # for Logistic Regression algorithm

from sklearn.model_selection import train_test_split # to split the dataset for training and testing

from sklearn.model_selection import cross_validate

from sklearn.neighbors import KNeighborsClassifier  # for K nearest neighbours

from sklearn import svm  #for Support Vector Machine (SVM) Algorithm

from sklearn import metrics #for checking the model accuracy

from sklearn.tree import DecisionTreeClassifier
train, test = train_test_split(iris, test_size = 0.3)# in this our main data is split into train and test

# the attribute test_size=0.3 splits the data into 70% and 30% ratio. train=70% and test=30%

print(train.shape)

print(test.shape)
train_X = train[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm',

                 'PetalWidthCm']]  # taking the training data features

train_y = train.Species  # output of our training data

test_X = test[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm',

                 'PetalWidthCm']]  # taking test data features

test_y = test.Species  #output value of test data
model = svm.SVC()  #select the algorithm

model.fit(

    train_X, train_y

)  # we train the algorithm with the training data and the training output

prediction = model.predict(

    test_X)  #now we pass the testing data to the trained algorithm

print('The accuracy of the SVM is:', metrics.accuracy_score(

    prediction, test_y))  #now we check the accuracy of the algorithm.

#we pass the predicted output by the model and the actual output
model = LogisticRegression()

model.fit(train_X, train_y)

prediction = model.predict(test_X)

print('The accuracy of the Logistic Regression is',

      metrics.accuracy_score(prediction, test_y))
model=DecisionTreeClassifier()

model.fit(train_X,train_y)

prediction=model.predict(test_X)

print('The accuracy of the Decision Tree is',metrics.accuracy_score(prediction,test_y))

print(metrics.classification_report(test_y, prediction))

print(metrics.confusion_matrix(test_y, prediction))
model=KNeighborsClassifier(n_neighbors=6) #this examines 3 neighbours for putting the new data into a class

model.fit(train_X,train_y)

prediction=model.predict(test_X)

print('The accuracy of the KNN is',metrics.accuracy_score(prediction,test_y))
a_index=list(range(1,11))

a=pd.Series()

x=[1,2,3,4,5,6,7,8,9,10]

for i in list(range(1,11)):

    model=KNeighborsClassifier(n_neighbors=i) 

    model.fit(train_X,train_y)

    prediction=model.predict(test_X)

    a=a.append(pd.Series(metrics.accuracy_score(prediction,test_y)))

plt.plot(a_index, a)

plt.xticks(x)