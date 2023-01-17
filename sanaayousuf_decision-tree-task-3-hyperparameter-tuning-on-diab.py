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
import math

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

df=pd.read_csv("/kaggle/input/pima-indians-diabetes-database/diabetes.csv")
df.head()
df.shape
X=df.iloc[:,:8].values

Y=df.iloc[:,8].values
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
X=scaler.fit_transform(X)

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state = 1,test_size=0.2)
from sklearn.tree import DecisionTreeClassifier

cnn=DecisionTreeClassifier()

cnn.fit(X_train,Y_train)


Y_pred=cnn.predict(X_test)
from sklearn.metrics import accuracy_score

accuracy_score(Y_test,Y_pred)
param_dict={

    'criterion':['gini','entropy'],

    'max_depth':[1,2,3,4],

    'max_features':[2,3,4,5],

    'splitter':['best', 'random'],

    

}
from sklearn.model_selection import GridSearchCV

grid=GridSearchCV(cnn,param_grid=param_dict,cv=10)
grid.fit(X_train,Y_train)
grid.estimator
grid.best_score_
grid.best_params_