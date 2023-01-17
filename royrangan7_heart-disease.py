# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import tree, metrics, model_selection, preprocessing

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
heartData = pd.read_csv("/kaggle/input/"+filename)

heartData.head(10)

#heartData['target'].isnull().sum()

X_train, X_test, y_train, y_test = model_selection.train_test_split(heartData.drop('target', 1), heartData['target'], test_size = .3, random_state=10)

# Decision Tree Classifier

decisionTreeClassifier = DecisionTreeClassifier()

decisionTreeClassifier.fit(X_train,y_train)

predict_heart_data = decisionTreeClassifier.predict(X_test)



# How did our model perform?

count_misclassified = (y_test != predict_heart_data).sum()

print('Decision Tree Classifier Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, predict_heart_data)

print('Decision Tree Classifier Accuracy: {:.2f}'.format(accuracy))



# Random Forest Classifier

randomForestClassifier = RandomForestClassifier(max_depth=5)

randomForestClassifier.fit(X_train, y_train)

predict_heart_data = randomForestClassifier.predict(X_test)



# How did our model perform?

count_misclassified = (y_test != predict_heart_data).sum()

print('Random Forest Classifier Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, predict_heart_data)

print('Random Forest Classifier Accuracy: {:.2f}'.format(accuracy))
