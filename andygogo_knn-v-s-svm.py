# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

%matplotlib inline
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns

#load the dataset 
iris = pd.read_csv("../input/Iris.csv")
iris.head()

#visulize the dataset
iris.plot(kind="scatter",x="SepalLengthCm", y='SepalWidthCm',s=50, figsize=(7,7))
g = sns.FacetGrid(iris,hue='Species',size=7)\
.map(plt.scatter, "SepalLengthCm","SepalWidthCm",s=70)\
.add_legend()
g = sns.FacetGrid(iris,hue='Species',size=7)\
.map(plt.scatter, "PetalLengthCm","PetalWidthCm",s=70)\
.add_legend()
sns.FacetGrid(iris,hue="Species",size=7)\
.map(sns.kdeplot,"PetalLengthCm").add_legend()
#train knn classifier

import random
from sklearn.neighbors import KNeighborsClassifier 
#choose test dataset randomly
arr = np.arange(150)
random.shuffle(arr)

ix_train = arr[:120]
ix_test = arr[120:]
#two features are selected to train the knn model
trainset=iris[['PetalLengthCm','PetalWidthCm','Species']].iloc[ix_train]
testset = iris[['PetalLengthCm','PetalWidthCm','Species']].iloc[ix_test]

#prepare the input
X_train = trainset.values[:,:2]
y_train = trainset.values[:,-1].flatten()
X_test = testset.values[:,:2]
y_test = testset.values[:,-1].flatten()

knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train,y_train)

knn.score(X_test,y_test)
#svm vs knn(four features here, it seems like there is 
#no difference whether we use only two features or all four of them in knn model)
from sklearn import svm
from sklearn.cross_validation import train_test_split

data = iris.drop(['Id'],axis=1).values
X = data[:,:-1]
y = data[:,-1]
#randomly choose the test dataset
X_train, X_test,y_train,y_test = train_test_split(X, y, test_size = 0.3)

clf = svm.SVC(kernel='rbf',decision_function_shape='ovr')

clf.fit(X_train,y_train)

#prediction = clf.predict(X_test)
#n = prediction.size
#r = 0.0
#for i in range(n):
#    if prediction[i]==y_test[i]:
#        r=r+1
#print("The accuracy is: %d%%" %(r*100/n) )

print(clf.score(X_test,y_test))

knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train,y_train)

print(knn.score(X_test,y_test))