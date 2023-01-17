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



train_df=pd.read_csv("/kaggle/input/titanic/train.csv")

test_df=pd.read_csv("/kaggle/input/titanic/test.csv")
# let us try automatic EDA by autoviz

import pandas_profiling # library for automatic EDA

from autoviz.AutoViz_Class import AutoViz_Class
AV = AutoViz_Class()

report=AV.AutoViz("/kaggle/input/titanic/train.csv")
train_df.head()
train_df.info()
train_df.describe()
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
train_df["Survived"].value_counts()/len(train_df)
# We are dropping Cabin as it misses more than 50 percent values.

train_data=train_df.drop("Cabin",axis=1)
train_df["Pclass"].value_counts().sort_values(ascending=False)/len(train_df)    # most of the passengers are from Pclass 3
train_df["Sex"].value_counts()/len(train_df)          # majority is male
train_df["Embarked"].value_counts()/len(train_df)     # most of them embarked from S
# Add some relevant features from the list of features

train_data["Relatives"]=train_data["SibSp"]+train_data["Parch"]

train_data[["Relatives","Survived"]].groupby(["Relatives"]).mean()

train_data["Fare"]=pd.cut(train_data["Fare"],bins=[0,150,300,520],labels=[1,2,3])

train_data["Fare"].value_counts()
train_data["Age"]=pd.cut(train_data["Age"],bins=[0,15,30,45,60,80],labels=[1,2,3,4,5])

train_data["Age"].value_counts()
# Let's groupby Sex with Survived

train_data[["Sex","Survived"]].groupby(["Sex"]).mean()   # This proves the fact that more females survived in the mishap
train_data[["Pclass","Survived"]].groupby(["Pclass"]).mean()  # This also sounds right.
train_data=train_data.drop(["SibSp","Parch"],axis=1)

train_data.head()
from sklearn.base import TransformerMixin,BaseEstimator



class DataFrameSelector(TransformerMixin,BaseEstimator):

    def __init__(self,attribute_names):

        self.attribute_names=attribute_names

    def fit(self,X,y=None):

        return self

    def transform(self,X):

        return X[self.attribute_names]
from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline



num_pipeline=Pipeline([("select_numeric",DataFrameSelector(["Relatives"])),

                      ("imputer",SimpleImputer(strategy="median"))])

num_pipeline.fit_transform(train_data)
from sklearn.preprocessing import OneHotEncoder



cat_pipeline=Pipeline([("select_cat",DataFrameSelector(["Pclass","Age","Sex","Fare","Embarked"])),

                      ("imputer",SimpleImputer(strategy="most_frequent")),

                      ("onehotenc",OneHotEncoder(sparse=False))])
cat_pipeline.fit_transform(train_data)
from sklearn.pipeline import FeatureUnion



combined_pipeline=FeatureUnion([("num_pipeline",num_pipeline),("cat_pipeline",cat_pipeline)])
X_train=combined_pipeline.fit_transform(train_data)
y_train=train_data["Survived"]
# Try some models

from sklearn.svm import SVC



svc_clf=SVC(kernel='rbf',gamma="auto")

svc_clf.fit(X_train,y_train)
test_data=test_df.drop("Cabin",axis=1)



test_data["Relatives"]=test_data["SibSp"]+test_data["Parch"]

test_data=test_data.drop(["SibSp","Parch"],axis=1)
test_data["Fare"]=pd.cut(test_data["Fare"],bins=[0,150,300,1000],labels=[1,2,3])

#test_data["Fare"].value_counts()
test_data["Fare"]
test_data["Age"]=pd.cut(test_data["Age"],bins=[0,15,30,45,60,100],labels=[1,2,3,4,5])

#test_data["Age"].value_counts()
test_data["Age"]
test_data.head()
X_test=combined_pipeline.transform(test_data)
from sklearn.model_selection import cross_val_predict

from sklearn.metrics import f1_score



y_train_pred=cross_val_predict(svc_clf,X_train,y_train,cv=5)

svc_acc=sum(y_train==y_train_pred)/len(train_data)

svc_acc     
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier



param_grid={"n_estimators":[10,30,100,300,1000],"max_features":['auto', 'sqrt', 'log2']}

forest_clf=RandomForestClassifier(random_state=42)

grid_search=GridSearchCV(forest_clf,param_grid,cv=5)

grid_search.fit(X_train,y_train)
grid_search.best_score_
grid_search.cv_results_["mean_test_score"]
grid_search.best_params_
from sklearn.neighbors import KNeighborsClassifier



param_grid_knn={"n_neighbors":[3,4,5,6,7],"weights":['uniform', 'distance'] }

knn_clf=KNeighborsClassifier()

grid_knn=GridSearchCV(knn_clf,param_grid_knn,cv=5)

grid_knn.fit(X_train,y_train)
grid_knn.best_score_
grid_knn.cv_results_["mean_test_score"]
grid_knn.best_params_
from sklearn.naive_bayes import GaussianNB

nb_clf=GaussianNB()

y_train_pred=cross_val_predict(nb_clf,X_train,y_train,cv=5)

nb_acc=sum(y_train==y_train_pred)/len(train_data)

nb_acc                                            # Performs well
from sklearn.linear_model import SGDClassifier

sgd_clf=SGDClassifier()

y_train_pred=cross_val_predict(sgd_clf,X_train,y_train,cv=5)

sgd_acc=sum(y_train==y_train_pred)/len(train_data)

sgd_acc                                                 # too bad
''''from sklearn.svm import SVC



svc_clf=SVC()

param_grid_svc={"kernel":['linear'],"C":[0.1,1,10,100,1000]}

grid_linearSVC=GridSearchCV(svc_clf,param_grid_svc,cv=5)

grid_linearSVC.fit(X_train,y_train)'''
from sklearn.svm import SVC

# to see if data is linearly or non-linearly separable

svc_clf_def=SVC(kernel='linear',C=100)

y_train_pred=cross_val_predict(svc_clf_def,X_train,y_train,cv=5)

svc_lin_acc=sum(y_train==y_train_pred)/len(train_data)

svc_lin_acc     
# Perceptron

from sklearn.linear_model import Perceptron

perceptron=Perceptron()

y_train_pred=cross_val_predict(perceptron,X_train,y_train,cv=5)

perceptron_acc=sum(y_train==y_train_pred)/len(train_data)

perceptron_acc 
from sklearn.ensemble import VotingClassifier



clf=VotingClassifier(estimators=[('svc', svc_clf), ('knn',knn_clf), ('sgd', sgd_clf),('rf',forest_clf)], voting='hard')

clf.fit(X_train,y_train)

y_train_pred=clf.predict(X_train)

voting_acc=sum(y_train==y_train_pred)/len(train_data)

voting_acc
y_pred=clf.predict(X_test)
output = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": y_pred

    })

output.to_csv('submission.csv',index=False)