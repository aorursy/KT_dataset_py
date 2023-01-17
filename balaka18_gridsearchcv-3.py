# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split,GridSearchCV

from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, confusion_matrix



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')

print(data.shape)

data.head()
data.info()
X = data.iloc[:,:8].values

Y = data.iloc[:,-1].values



scaler = StandardScaler()

scaler.fit_transform(X)
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=20,random_state = 3)

dt = DecisionTreeClassifier()

dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)

print("Accuracy score = ",accuracy_score(y_test,y_pred))

confusion_matrix(y_test,y_pred)
# Declare parameters dictionary

params_dict = {

    "criterion" : ['gini','entropy'],

    "max_depth" : [1,2,3,4,5,6,7,None]

}



gs = GridSearchCV(dt,param_grid = params_dict,cv = 10)

gs.fit(x_train,y_train)
# Finding the best parameter values and corresponding accuracy score

print(gs.best_params_)