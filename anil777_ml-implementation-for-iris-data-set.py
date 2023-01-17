import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))



import seaborn as sns



import matplotlib.pyplot as plt



# Loading data set 

iris = pd.read_csv("../input/Iris.csv")

iris.head()
iris.info() # To check if there is any inconsistency in the data
# Drop Id column because it is useless for us. 



iris.drop("Id", axis = 1, inplace = True)
iris.head()
sns.scatterplot(x="SepalLengthCm", y='SepalWidthCm', hue='Species', data = iris)
sns.scatterplot(x="PetalLengthCm", y='PetalWidthCm', hue='Species', data = iris)
iris.hist(edgecolor='black', linewidth=1.2)

fig=plt.gcf()

fig.set_size_inches(12,6)

plt.show()
# Now let us analyze about how Length and Width Vary for different Species. Idea here is to use violin plot. 



plt.figure(figsize=(15,10))

plt.subplot(2,2,1)

sns.violinplot(x='Species',y='PetalLengthCm',data=iris)

plt.subplot(2,2,2)

sns.violinplot(x='Species',y='PetalWidthCm',data=iris)

plt.subplot(2,2,3)

sns.violinplot(x='Species',y='SepalLengthCm',data=iris)

plt.subplot(2,2,4)

sns.violinplot(x='Species',y='SepalWidthCm',data=iris)
import sklearn

from sklearn.linear_model import LogisticRegression  # for Logistic Regression algorithm

from sklearn.model_selection import train_test_split #to split the dataset for training and testing

from sklearn.neighbors import KNeighborsClassifier  # for K nearest neighbours

from sklearn import svm  #for Support Vector Machine (SVM) Algorithm

from sklearn import metrics #for checking the model accuracy

from sklearn.tree import DecisionTreeClassifier #for using Decision Tree Algoithm

iris.shape
plt.figure(figsize=(7,4)) 

sns.heatmap(iris.corr(),annot=True) 

plt.show()
train, test = train_test_split(iris, test_size = 0.3)

print(train.shape)

print(test.shape)
train_X = train[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]

train_y=train.Species

test_X= test[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']] 

test_y =test.Species
train_X.head(2)
test_X.head(2)
train_y.head()
model = svm.SVC()

model.fit(train_X,train_y)  # Training the Algorithm 