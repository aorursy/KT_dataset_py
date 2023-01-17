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
from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
iris_df=load_iris()
iris_df=load_iris()
X_train,X_test,y_train,y_test=train_test_split(iris_df.data,iris_df.target,test_size=0.3,random_state=0)


from sklearn.model_selection import RandomizedSearchCV
# Create a pipeline

pipe = Pipeline([("classifier", LogisticRegression())])

# Create dictionary with candidate learning algorithms and their hyperparameters

grid_param = [

                {"classifier": [LogisticRegression()],

                 "classifier__penalty": ['l2','l1'],

                 "classifier__C": np.logspace(0, 4, 10)

                 },

                {"classifier": [LogisticRegression()],

                 "classifier__penalty": ['l2'],

                 "classifier__C": np.logspace(0, 4, 10),

                 "classifier__solver":['newton-cg','saga','sag','liblinear'] ##This solvers don't allow L1 penalty

                 },

                {"classifier": [RandomForestClassifier()],

                 "classifier__n_estimators": [10, 100, 1000],

                 "classifier__max_depth":[5,8,15,25,30,None],

                 "classifier__min_samples_leaf":[1,2,5,10,15,100],

                 "classifier__max_leaf_nodes": [2, 5,10]}]

# create a gridsearch of the pipeline, the fit the best model

gridsearch = RandomizedSearchCV(pipe, grid_param, cv=5, verbose=0,n_jobs=-1) # Fit grid search

best_model = gridsearch.fit(X_train,y_train)
print(best_model.best_estimator_)

print("The mean accuracy of the model is:",best_model.score(X_test,y_test))