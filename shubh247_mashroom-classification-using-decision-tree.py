import numpy as np # linear algebra

import pandas as pd # data processing

import time

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import Imputer

from sklearn.metrics import confusion_matrix

from subprocess import check_output

from sklearn import tree

from IPython.display import Image as PImage

from subprocess import check_call

from PIL import Image, ImageDraw, ImageFont

import re

import matplotlib.pyplot as plt
#read dataset

data = pd.read_csv('../input/mushrooms.csv')

   # chose path where .data file present
data.head()  #to find first 5 values
data.columns 
data.info()
data.describe()
data['stalk-root'].value_counts(dropna=False)
#data["stalk-root"].replace(["?"], ["b"], inplace= True)   # use b(it is mode) 
from sklearn.preprocessing import LabelEncoder



labelencoder=LabelEncoder()

for column in data.columns:

    data[column] = labelencoder.fit_transform(data[column])
data['stalk-root'].value_counts(dropna=False)
data['stalk-root'].describe()
data["stalk-root"] = data["stalk-root"].astype(object)
#data["stalk-root"][::].replace(0, 1.109565) 

#1.109565 = mean 
data["stalk-root"] = data["stalk-root"].astype(int)
import seaborn as sns

plt.figure(figsize=(16,10))

sns.heatmap(data.corr(), annot=True);
data = data.drop('veil-type', axis=1)

data=data.drop(["stalk-root"],axis=1)
data.head()
data.info()
X = data.drop(['class'], axis=1)  #remove target from train dataset

y = data['class'] # test dataset with target 
 # divide dataset into 50% train, and other 50% test.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
clf1 = DecisionTreeClassifier(criterion = "gini",                # model design 

            random_state = 100,max_depth=2, min_samples_leaf=5, ) 

# split dataset into depth 2(0,1,2)

# stop dataset when leaf is 5 . 
clf1 = clf1.fit(X_train, y_train)  #training the model 
y_pred = clf1.predict(X_test)  # prediction on test dataset 
print('accuracy of train dataset is',clf1.score(X_train, y_train))
print('accuracy of test dataset is',clf1.score(X_test, y_test))
from sklearn.metrics import classification_report

print("Decision tree Classification report", classification_report(y_test, y_pred))
confusion_matrix(y_test, y_pred)
cfm=confusion_matrix(y_test, y_pred)

sns.heatmap(cfm, annot=True, linewidth=5, cbar=None)

plt.title('Decision Tree Classifier confusion matrix')

plt.ylabel('True label')

plt.xlabel('predicted label')
import graphviz

import os

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'
from sklearn.tree import DecisionTreeClassifier, export_graphviz

import graphviz

dot_data = export_graphviz(clf1, out_file='tree1.dot',

                          feature_names=X.columns,

                          filled=True, rounded = True, 

                          special_characters= True,

             class_names=['0','1']  )

graph = graphviz.Source(dot_data)



os.system('dot -Tpng tree1.dot -o tree1.png')
from IPython.display import Image

Image(filename="tree1.png", height=1000, width=1000)
clf2 = DecisionTreeClassifier(criterion = "gini", 

            random_state = 100,max_depth=5, min_samples_leaf=15, ) 

clf2 = clf2.fit(X_train, y_train)   

# split dataset into depth 5

# stop dataset when leaf is 15. 
y_pred = clf2.predict(X_test)
print('accuracy of train dataset is',clf2.score(X_train, y_train))
print('accuracy of test dataset is',clf2.score(X_test, y_test))
from sklearn.metrics import classification_report

print("Decision tree Classification report", classification_report(y_test, y_pred))
confusion_matrix(y_test, y_pred)
cfm=confusion_matrix(y_test, y_pred)

sns.heatmap(cfm, annot=True, linewidth=5, cbar=None)

plt.title('Decision Tree Classifier confusion matrix')

plt.ylabel('True label')

plt.xlabel('predicted label')
import graphviz

import os

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'
from sklearn.tree import DecisionTreeClassifier, export_graphviz

import graphviz

dot_data = export_graphviz(clf2, out_file='tree2.dot',

                          feature_names=X.columns,

                          filled=True, rounded = True, 

                          special_characters= True,

             class_names=['0','1']  )

graph = graphviz.Source(dot_data)



os.system('dot -Tpng tree2.dot -o tree2.png')
from IPython.display import Image

Image(filename="tree2.png", height=1000, width=1000)
clf3 = DecisionTreeClassifier( 

            criterion = "entropy", random_state = 100, 

            max_depth = 3, min_samples_leaf = 10) 

clf3 = clf3.fit(X_train, y_train)

# split dataset into depth 3(0,1,2,3)

# stop dataset when leaf is 10 . 
y_pred = clf3.predict(X_test)
print('accuracy of train dataset is',clf3.score(X_train, y_train))
print('accuracy of test dataset is',clf3.score(X_test, y_test))
print("Decision tree Classification report", classification_report(y_test, y_pred))
confusion_matrix(y_test, y_pred)
cfm=confusion_matrix(y_test, y_pred)

sns.heatmap(cfm, annot=True, linewidth=5, cbar=None)

plt.title('Decision Tree Classifier confusion matrix')

plt.ylabel('True label')

plt.xlabel('predicted label')
import graphviz

import os

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'
from sklearn.tree import DecisionTreeClassifier, export_graphviz

import graphviz

dot_data = export_graphviz(clf3, out_file='tree3.dot',

                          feature_names=X.columns,

                          filled=True, rounded = True, 

                          special_characters= True,

             class_names=['0','1']  )

graph = graphviz.Source(dot_data)



os.system('dot -Tpng tree3.dot -o tree3.png')
from IPython.display import Image

Image(filename="tree3.png", height=1000, width=1000)
clf4 = DecisionTreeClassifier( 

            criterion = "entropy", random_state = 100, 

            max_depth = 10, min_samples_leaf = 20) 

clf4 = clf4.fit(X_train, y_train)

# split dataset into depth 3(0,1,2,3)

# stop dataset when leaf is 10 . 
y_pred = clf4.predict(X_test)
print('accuracy of train dataset is',clf4.score(X_train, y_train))
print('accuracy of test dataset is',clf4.score(X_test, y_test))
print("Decision tree Classification report", classification_report(y_test, y_pred))
confusion_matrix(y_test, y_pred)
cfm=confusion_matrix(y_test, y_pred)

sns.heatmap(cfm, annot=True, linewidth=5, cbar=None)

plt.title('Decision Tree Classifier confusion matrix')

plt.ylabel('True label')

plt.xlabel('predicted label')
import graphviz

import os

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'
from sklearn.tree import DecisionTreeClassifier, export_graphviz

import graphviz

dot_data = export_graphviz(clf4, out_file='tree4.dot',

                          feature_names=X.columns,

                          filled=True, rounded = True, 

                          special_characters= True,

             class_names=['0','1']  )

graph = graphviz.Source(dot_data)



os.system('dot -Tpng tree4.dot -o tree4.png')
from IPython.display import Image

Image(filename="tree4.png", height=1000, width=1000)