# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd



#grafs

import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns





# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier



# Import Dependencies

%matplotlib inline



# Start Python Imports

import math, time, random, datetime



# Data Manipulation

import numpy as np

import pandas as pd



# Visualization 

import matplotlib.pyplot as plt

import missingno

import seaborn as sns

plt.style.use('seaborn-whitegrid')



# Preprocessing

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize



# Machine learning

import catboost

from sklearn.model_selection import train_test_split

from sklearn import model_selection, tree, preprocessing, metrics, linear_model

from sklearn.svm import LinearSVC

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from catboost import CatBoostClassifier, Pool, cv



# Let's be rebels and ignore warnings for now

import warnings

warnings.filterwarnings('ignore')
train=pd.read_csv("/kaggle/input/titanic/train.csv")



X_test=pd.read_csv("/kaggle/input/titanic/test.csv")

X_test.copy()

train.info()
train
# train.columns
from fastai.tabular import *
procs = [FillMissing, Categorify, Normalize]



dep_var = 'Survived'

cat_names = ['Sex', 'Cabin', 'Embarked']

cont_names = ['PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'SibSp']

X=train[cont_names]

X=train[cat_names]
data = (TabularList.from_df(train, procs=procs, cont_names=cont_names, cat_names=cat_names)

        .split_by_idx(valid_idx=range(int(len(train)*0.9),len(train)))

        .label_from_df(cols=dep_var)

        .add_test(TabularList.from_df(X_test, cat_names=cat_names, cont_names=cont_names, procs=procs))

        .databunch())

print(data.train_ds.cont_names)

print(data.train_ds.cat_names)
learn = tabular_learner(data, layers=[1000,500], metrics=accuracy, wd=0.1)
learn.fit_one_cycle(5, 2.5e-2)
learn.fit_one_cycle(5, 2.6e-9)
# learn.fit_one_cycle(5, 2.6e-9)
learn.save('stage-1') #save the model
preds, _ = learn.get_preds(ds_type=DatasetType.Test)

pred_prob, pred_class = preds.max(1)
submission = pd.DataFrame({'PassengerId': X_test.PassengerId, 'Survived':pred_class})
submission.to_csv('my_submission.csv', index=False)