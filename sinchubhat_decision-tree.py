import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import sklearn.datasets as datasets

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
iris=datasets.load_iris()
iris
type(iris)
iris.keys()
iris['data']
iris['target']
iris['frame']
iris['target_names']
iris['DESCR']
iris['feature_names']
iris['filename']
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df.head()
df.tail()
df.shape
df.size
# Check the column names

df.columns
#Check null values

df.isnull().sum()
df.describe()
y=iris.target
y
# Plot histogram of the given data 

df.hist(figsize = (12,12))
# Pairplot of the given data

sns.pairplot(df)
# import

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
from sklearn .model_selection import train_test_split 

x_train,x_test,y_train,y_test = train_test_split(df,y,test_size = 0.30,random_state = 42)
# Defining the decision tree algorithm

dtree=DecisionTreeClassifier()

dtree.fit(x_train,y_train)
# Accuracy For training data

predict = dtree.predict(x_train)

print("Accuracy of training data : ",accuracy_score(predict,y_train)*100,"%")

print("Confusion matrix of training data :'\n' ",confusion_matrix(predict,y_train))

sns.heatmap(confusion_matrix(predict,y_train),annot = True,cmap = 'BuGn')
# Accuracy For testing data

predict = dtree.predict(x_test)

print("Accuracy of testing data : ",accuracy_score(predict,y_test)*100,"%")

print("Classification Report : ",classification_report(predict,y_test))

print("Confusin matrix of testing data :\n ",confusion_matrix(predict,y_test))

sns.heatmap(confusion_matrix(predict,y_test),annot = True,cmap = 'BuGn')
from sklearn import tree

plt.figure(figsize  = (19,19))

tree.plot_tree(dtree,filled = True,rounded = True,proportion = True,node_ids = True , feature_names = iris.feature_names)

plt.show()