import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.datasets import load_wine

wine=load_wine()
#Features dataframe

df=pd.DataFrame(wine.data,columns=wine.feature_names)

df.head()

X=df
X.isnull().sum()
#label

y=pd.Series(wine.target,name="Target")

wine.target_names
#feature selection

from sklearn.feature_selection import SelectKBest,chi2

sk=SelectKBest(chi2,k=10)

sk.fit(X,y)

sk.transform(X)
sk.pvalues_
#get the column names to select

X.columns
X_data =X[["alcohol","ash","nonflavanoid_phenols","hue","proline"]] #selecting best features
#splitting the data based on train test and split

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X_data,y,test_size=0.2,random_state=11)
#pipeline  needs preprocessing and models

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import ExtraTreesClassifier

my_pipeline=make_pipeline(StandardScaler(),ExtraTreesClassifier())
my_pipeline.fit(X_train,y_train)
y_pred=my_pipeline.predict(X_test)

y_pred
my_pipeline.score(X_test,y_test)
#checking the accuracy of model

from sklearn.metrics import accuracy_score

accuracy_score(y_test,y_pred)