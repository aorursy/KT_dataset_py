!pip install pydotplus

!pip install graphviz

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn as sk # for decision tree mapping

import seaborn as sns# for corrplot



from io import StringIO

from sklearn import preprocessing

from sklearn import tree

from sklearn.tree import export_graphviz



import pydotplus

from sklearn.preprocessing import StandardScaler







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
my_data = pd.read_csv("../input/heart-disease-uci/heart.csv")
my_data.head
my_data.describe()
correlation_data = my_data.corr()

correlation_data
#for more visual using seaborn 



display = sns.heatmap(correlation_data)
#transforming into dummy data

transform_data = pd.get_dummies(my_data, columns = ['sex','cp','fbs','restecg','exang','slope','ca','thal'])



transform_data #added new cols
#didnt add the Feature Scaling since it is DS tree nor RF or Xgboost. It is needed only on Linear Regression / KNN / kmeans 
#creating the feature variable and the independent variable

from sklearn.model_selection import train_test_split #for testing training

from sklearn.tree import DecisionTreeClassifier # for decision tree

from sklearn import metrics #for accuracy 



y = transform_data.target # TAGRET #added this parameter because it returns error 

X = transform_data.drop(['target'], axis = 1) # FEATURES 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)# 70% training and 30% test
clf = DecisionTreeClassifier() #call decision tree classifier



clf = clf.fit(X_train,y_train) #training decision tree classifier



y_pred = clf.predict(X_test) #result of the training
# Accuracy

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#optimizing decision tree

clf = DecisionTreeClassifier(criterion="entropy", max_depth = 3) 

#criterion : optional (default=”gini”) or Choose attribute selection measure: This parameter allows us to use the different-different attribute selection measure. Supported criteria are “gini” for the Gini index and “entropy” for the information gain

#splitter : string, optional (default=”best”) or Split Strategy: This parameter allows us to choose the split strategy. Supported strategies are “best” to choose the best split and “random” to choose the best random split.

#max_depth : int or None, optional (default=None) or Maximum Depth of a Tree: The maximum depth of the tree. If None, then nodes are expanded until all the leaves contain less than min_samples_split samples. The higher value of maximum depth causes overfitting, and a lower value causes underfitting
# Train Decision Tree Classifer

clf = clf.fit(X_train,y_train)



#Predict the response for test dataset

y_pred = clf.predict(X_test)



# Model Accuracy, how often is the classifier correct?

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

X.shape
from sklearn import tree

from io import StringIO

from IPython.display import Image

dot_data = StringIO()

tree.export_graphviz(clf, out_file=dot_data, class_names = str(y) )



graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

graph.write_png('heart stroke.png')

Image(graph.create_png())

#if SEX is Female and AGE is Lesser than 57, it would create another split that if the AGE is less than 46; 30 of them doesnt have heart disease -- Female 46YO doesnt have heart attack and 46 above is unlikely to have heart attack

#if SEX is Female and AGE is Greater than 57, it would create a split that if the individual female 12 of the doesnt have heart disease and if male 16 of them do -- If the a female is above 57 years old still it is unlikely to have heart disease compared to Male that is above 57 years old



#if Sex is Male rest eCG 0 and would create another split that if oldpeak is above 1 it is likely to have a heart attack
#Take away

#Female 46YO doesnt have heart attack and 46 above is unlikely to have heart attack

#If the a female is above 57 years old still it is unlikely to have heart disease compared to Male that is above 57 years old

#If the individual is male and resting electrocardiographic results is below 0.5 (0-1) but ST depression above 0.5 (0-6.2) it is very likely to have heart disease



X.columns.tolist()