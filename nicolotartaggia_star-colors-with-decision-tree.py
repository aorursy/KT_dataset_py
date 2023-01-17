# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier

from sklearn.model_selection import train_test_split # Import train_test_split function

from sklearn import metrics # Import scikit-learn metrics module for accuracy calculation

from sklearn import preprocessing # Import preprocessing for String-Int conversion

from sklearn import tree

import graphviz





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#import dataset

stars_dataset = pd.read_csv("/kaggle/input/star-dataset/6 class csv.csv")

stars_dataset.head()
# select features

features = stars_dataset.drop('Star color', axis=1)

# select target

target = stars_dataset['Star color']



# convert 'Star color' and 'Spectral Class' values from String to Int using LabelEncoder

features['Spectral Class'] = preprocessing.LabelEncoder().fit_transform(features['Spectral Class'])



# split dataset into training set and test set

x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=1) # 70% training and 30% test



# Create Decision Tree classifer object with these parameters

clf = DecisionTreeClassifier(criterion = 'entropy',max_depth = 3)

# Train Decision Tree Classifer

clf = clf.fit(x_train,y_train)

# Predict the response for test dataset

y_pred = clf.predict(x_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# print the tree and see the result

dot_data = tree.export_graphviz(clf, out_file=None, rounded=True, feature_names = features.columns, class_names = list(set(target)), filled = True) 

graph = graphviz.Source(dot_data) 

graph