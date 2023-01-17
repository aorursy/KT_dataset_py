!pip install pydotplus
import pydotplus
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn.neighbors import KNeighborsClassifier



from sklearn.tree import DecisionTreeClassifier



from sklearn import preprocessing



from sklearn.model_selection import train_test_split



from sklearn import metrics

import matplotlib.pyplot as plt



from sklearn.externals.six import StringIO

import graphviz

import matplotlib.image as mpimg

from sklearn import tree

%matplotlib inline 

irisclass=pd.read_csv('/kaggle/input/iris/Iris.csv')
irisclass.head()
irisclass.shape
irisclass.dtypes
irisclass.drop('Id', axis=1, inplace=True)
irisclass.head(2)
irisclass['Species'].value_counts()
fig = irisclass[irisclass.Species=='Iris-setosa'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='magenta', label='Setosa')

irisclass[irisclass.Species=='Iris-versicolor'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='cyan', label='versicolor',ax=fig) 

irisclass[irisclass.Species=='Iris-virginica'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='green', label='virginica', ax=fig)

fig.set_xlabel("Sepal Length")

fig.set_ylabel("Sepal Width")

fig.set_title("Sepal Length VS Width")

fig=plt.gcf()  #Get the current figure. If no current figure exists, a new one is created using figure().

fig.set_size_inches(10,6)

plt.show()
fig = irisclass[irisclass.Species=='Iris-setosa'].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',color='magenta', label='Setosa')

irisclass[irisclass.Species=='Iris-versicolor'].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',color='cyan', label='versicolor',ax=fig) 

irisclass[irisclass.Species=='Iris-virginica'].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',color='green', label='virginica', ax=fig)

fig.set_xlabel("Petal Length")

fig.set_ylabel("Petal Width")

fig.set_title("Petal Length VS Width")

fig=plt.gcf()  #Get the current figure. If no current figure exists, a new one is created using figure().

fig.set_size_inches(10,6)

plt.show()
X=irisclass[['PetalLengthCm', 'PetalWidthCm']]

X[0:2]
y=irisclass[['Species']]

y[0:2]
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)
X_trainset.shape
y_trainset.shape
X_testset.shape
y_testset.shape
irisnotatree= DecisionTreeClassifier(criterion="entropy", max_depth = None)

irisnotatree
irisnotatree.fit(X_trainset, y_trainset)
prediris=irisnotatree.predict(X_testset)
B=irisclass[['SepalLengthCm', 'SepalWidthCm']]

B[0:2]
v=irisclass[['Species']]

v[0:2]
B_trainset, B_testset, v_trainset, v_testset = train_test_split(B, v, test_size=0.3, random_state=3)
B_trainset.shape
B_testset.shape
v_trainset.shape
v_testset.shape
sepaltree= DecisionTreeClassifier(criterion="entropy", max_depth = None)
sepaltree.fit(B_trainset, v_trainset)
stree= sepaltree.predict(B_testset)
print("DecisionTrees's Sepal Accuracy: ", metrics.accuracy_score(v_testset, stree))

print("DecisionTrees's Petal Accuracy: ", metrics.accuracy_score(y_testset, prediris))
dot_data = StringIO()

filename = "iris.png"

featureNames = irisclass.columns[2:4]

targetNames = irisclass["Species"].unique().tolist()

out=tree.export_graphviz(irisnotatree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_trainset), filled=True,  special_characters=True,rotate=False)  

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

graph.write_png(filename)

img = mpimg.imread(filename)

plt.figure(figsize=(100, 200))

plt.imshow(img,interpolation='nearest')