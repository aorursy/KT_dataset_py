##### This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import pydotplus
import os
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.model_selection import train_test_split # (SciKit_Learn)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report # (Same as confusion matrix)
from numpy.linalg import eig
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score , classification_report 
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
sample_submission = pd.read_csv("../input/sample_submission.csv")
test = pd.read_csv("../input/test.csv")
train_xx , validate_xx = train_test_split(train , test_size = 0.15 , random_state = 100)
train_y =train_xx['label']
validate_y = validate_xx['label']
train_x = train_xx.drop('label', axis = 1)
validate_x = validate_xx.drop('label' , axis = 1)
model_dt = DecisionTreeClassifier(random_state = 100)
model_dt.fit(train_x, train_y)

validate_pred_dt =model_dt.predict(validate_x)
accuracy_score(validate_y, validate_pred_dt)
model_rf = RandomForestClassifier(random_state = 400 , n_estimators=400)
model_rf.fit(train_x,train_y)

validate_pred_rf = model_rf.predict(validate_x)
accuracy_score(validate_y , validate_pred_rf)

model_ab = AdaBoostClassifier(n_estimators=200 , random_state = 100)
model_ab.fit(train_x, train_y)

validate_pred_ab = model_ab.predict(validate_x)
accuracy_score(validate_y, validate_pred_ab)
from sklearn.neighbors import KNeighborsClassifier

model_knn_ = KNeighborsClassifier(n_neighbors=6)
model_knn_.fit(train_x,train_y)

validate_pred_knn = model_knn_.predict(validate_x)
accuracy_score(validate_y , validate_pred_knn)

### using KNN as it is giving Highest Accuracy ( But takes a lot of Time to Predict , so grab
### a coffee until then )
test_pred = model_knn_.predict(test)
pred_df_knn = pd.DataFrame({'ImageId' : test.index.values+1 , 'Label': test_pred})
pred_df_knn.to_csv('submission_1.csv' , index = False)
