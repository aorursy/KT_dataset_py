# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
print(train.shape, test.shape)
train.head()
import matplotlib.pyplot as plt
plt.figure(figsize = (10, 6))
plt.imshow(np.array(train.iloc[3,1:]).reshape(28, 28))
from sklearn.model_selection import train_test_split
train1, validate  = train_test_split(train, test_size = 0.3, random_state = 100)
train1_x = train1.iloc[:,1:]
train1_y = train1.iloc[:,0]
validate_x = validate.iloc[:,1:]
validate_y = validate.iloc[:,0]
print(train1_x.shape, train1_y.shape)
print(validate_x.shape, validate_y.shape)
#DecisionTree Classifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
DecisionTree = DecisionTreeClassifier(random_state = 100)
hyp_param = GridSearchCV(DecisionTree,{'max_depth':[10,20,30,40,50], 
                                                   'criterion':['gini', 'entropy']}, cv = 5)
hyp_param.fit(train1_x, train1_y)
print(hyp_param.best_params_)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
DecisionTree = DecisionTreeClassifier(max_depth = 20, criterion = 'entropy',random_state = 100)
DecisionTree.fit(train1_x, train1_y)
out_predict = DecisionTree.predict(validate_x)
print(accuracy_score(out_predict, validate_y))
#RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report

model_rf = RandomForestClassifier(random_state = 100, n_estimators = 300)
model_rf.fit(train1_x, train1_y)
model_rf_predict = model_rf.predict(validate_x)
print(accuracy_score(model_rf_predict, validate_y))
#AdaBoostClassifier

from sklearn.ensemble import AdaBoostClassifier
model_ada = AdaBoostClassifier(random_state = 100, n_estimators = 300)
model_ada.fit(train1_x, train1_y)
model_ada_predict = model_ada.predict(validate_x)
print(accuracy_score(model_ada_predict, validate_y))
#KNN Classifier

from sklearn.neighbors import KNeighborsClassifier
model_knn = KNeighborsClassifier(n_neighbors = 5)
model_knn.fit(train1_x, train1_y)
model_knn_predict = model_knn.predict(validate_x)
print(accuracy_score(model_knn_predict, validate_y))
#Predicting with the best model
output_predict = model_knn.predict(test)
#Writing Output to the CSV file
output = pd.DataFrame(columns = ['ImageId', 'Label'])
output['ImageId'] = range(1, 28001)
output['Label'] = output_predict
output.to_csv('Output.csv', index = False)
print('code running')