# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import pandas as pd

iris = pd.read_csv("../input/iris.csv")
iris.head()
iris['Species'].unique()
iris['Species'] = iris['Species'].map({'Iris-setosa':0,'Iris-versicolor':1, 'Iris-virginica':2})
iris.head() ## As you can see, all were converted
import matplotlib.pyplot as plt

import seaborn as sns
sns.heatmap(iris.corr(), annot=True) ##There's a great correlation between the length and width of sepals and petals in this dataset.
sns.scatterplot(iris['SepalLength'],iris['SepalWidth'], hue='Species', data=iris)
sns.scatterplot(iris['PetalLength'],iris['PetalWidth'], hue='Species', data=iris)
print(iris['Species'].value_counts()) ##Balanced dataset
features = ['SepalLength','SepalWidth','PetalLength','PetalWidth']
targets = ["Species"]
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(iris[features],iris[targets],test_size=0.20,random_state=0)
X_train.head()
Y_train.head()
from sklearn import svm
model = svm.SVC()
model.fit(X_train,Y_train)

predictions = model.predict(X_test)
predictions
print("The model precision was: ",round(model.score(X_test,predictions),2)*100,"%")