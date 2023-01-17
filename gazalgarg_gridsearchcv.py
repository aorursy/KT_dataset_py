# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

data=pd.read_csv("/kaggle/input/pima-indians-diabetes-database/diabetes.csv")
data
X=data.iloc[:,0:8]
Y=data.iloc[:,-1]
X
Y
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X=scaler.fit_transform(X)
X.shape
Y.shape
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=9)

X_train.shape
Y_train.shape
#Training the model using Decision Tree 
clf=DecisionTreeClassifier()
clf.fit(X_train,Y_train)


y_pred=clf.predict(X_test)

from sklearn.metrics import accuracy_score
acc=accuracy_score(Y_test,y_pred)
print("Accuracy is:",acc)
param_dist = {"max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None],
              "max_features": [1, 2, 3, 4, 5, 6, 7, 8, None],
              "max_leaf_nodes": [1, 2, 3, 4, 5, 6, None],
              "random_state": [1, 2, 3, 4, 5, 6,7,8,9, None],
              "criterion": ["gini", "entropy"]}
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(clf, param_grid=param_dist, cv=10, n_jobs=-1)

grid.fit(X_train,Y_train)

grid.best_estimator_
grid.best_score_
grid.best_params_