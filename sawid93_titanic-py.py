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
import sklearn

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV

from sklearn.pipeline import Pipeline

from sklearn import tree

from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold, RFE

from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

import numpy as np



from scipy.stats import uniform, randint

import seaborn as sns

import scikitplot as skplt

import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine

from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error

from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split



import xgboost as xgb
train=pd.read_csv('../input/train.csv')
train.head()
train.dtypes
train=train.drop(['Name','Ticket','Cabin'],axis=1)
train=pd.get_dummies(train, columns=['Sex','Embarked'])
train.head()
Features = train.columns

Features = Features.drop(['Survived'])

Features = Features.drop(['PassengerId'])
from sklearn.model_selection import StratifiedShuffleSplit

train['Age'] = train['Age'].fillna(train['Age'].mean())



df = train.fillna(0)



sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)



for train_index, test_index in sss.split(df[Features], df['Survived']):

  dftrain = df.iloc[train_index]

  dfval = df.iloc[test_index]
clf = Pipeline([

                  ('clf', RFE(DecisionTreeClassifier(min_samples_leaf = 100, max_depth = 5), 6))

                 ])



clf.fit(dftrain[Features], dftrain['Survived'])



p_train = clf.predict_proba(dftrain[Features])

print("1: auc train = " + str(roc_auc_score(dftrain['Survived'], p_train[:,1])))

p_val = clf.predict_proba(dfval[Features])

print("1: auc val = " + str(roc_auc_score(dfval['Survived'], p_val[:,1])))



#f = list(compress(Features, clf.named_steps['clf'].ranking_==1))

#print(f)
import matplotlib.pyplot as plt

import scikitplot as skplt

#skplt.metrics.plot_cumulative_gain(dfval['Survived'], p_val[:,1])

#plt.show()
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix, roc_curve,precision_recall_curve 

y_train_train = list(dftrain["Survived"].values)

select_X_train = dftrain.drop("Survived",axis = 1)

select_X_train = select_X_train.drop("PassengerId",axis = 1)

selection_model =  XGBClassifier()

selection_model.fit(select_X_train, y_train_train)

select_X_train.shape
val_Y = list(dftrain["Survived"].values)

val_X = dftrain.drop("Survived",axis = 1)

val_X =val_X.drop("PassengerId",axis = 1)

probabilities = selection_model.predict_proba(val_X)

probab_val = probabilities[:,1]

score = roc_auc_score(val_Y,probab_val)

print(score)
val_Y = list(dfval["Survived"].values)

val_X = dfval.drop("Survived",axis = 1)

val_X =val_X.drop("PassengerId",axis = 1)

probabilities = selection_model.predict_proba(val_X)

probab_val = probabilities[:,1]

score = roc_auc_score(val_Y,probab_val)

print(score)