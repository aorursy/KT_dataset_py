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
df=pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
df=df.astype('int8')

df
x=df.iloc[:,0:-1].values

y=df.iloc[:,-1].values
clf=DecisionTreeClassifier()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
clf.fit(x_train,y_train)

y_pred=clf.predict(x_test)
accuracy_before_cv=accuracy_score(y_test,y_pred)
param_dist={

    "max_depth":[1,2,3,4,5,None],

    "criterion":["gini","entropy"],

    "min_samples_leaf":[1,2,3,4,5,None],

    "min_samples_split":[1,2,3,4,5,None]

}
from sklearn.model_selection import GridSearchCV

grid=GridSearchCV(clf,param_grid=param_dist,cv=10,n_jobs=-1)
grid.fit(x_train,y_train)
accuracy_after_cv=grid.best_score_
print("ACCURACY SCORE before using hyparameter optimisation ",accuracy_before_cv*100)

print("ACCURACY SCORE after using hyparameter optimisation",accuracy_after_cv*100)
print("Optimal Parameters are: ",grid.best_params_)