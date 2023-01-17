import pandas as pd

import numpy as np



data = pd.read_csv("../input/mushrooms.csv")



data.head(5)
data.info()
from sklearn import preprocessing



le = preprocessing.LabelEncoder()



data.columns
data.shape
from sklearn import preprocessing



le = preprocessing.LabelEncoder()



for col in data.columns:

    data[col] = le.fit_transform(data[col])

   
data.head(5)
from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size = 0.2) 



train_y = train['class']

train_x = train[[x for x in train.columns if 'class' not in x ]]



test_y = train['class']

test_x = train[[x for x in train.columns if 'class' not in x ]]
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier

import xgboost

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from matplotlib import pyplot as plt

import seaborn as sns

sns.set(font_scale = 1.5)

%matplotlib inline



models = [SVC(kernel='rbf', random_state=0), 

          SVC(kernel='linear', random_state=0), 

          XGBClassifier(), 

          LogisticRegression(),

         RandomForestClassifier(n_estimators= 10, max_depth= 10, random_state= 0)]



model_names = ['SVC_rbf', 'SVC_linear', 'xgboost', 'Logistic Regression', 'RandomForest']



for i, model in enumerate(models):

    model.fit(train_x, train_y)

    print ('The accurancy of ' + model_names[i] + ' is ' + 

           str(accuracy_score(test_y, model.predict(test_x))) )

ax= xgboost.plot_importance(models[2])