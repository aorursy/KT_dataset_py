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
import pandas as pd

import numpy as np

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegressionCV

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.linear_model import LinearRegression

import seaborn as sns

from sklearn import metrics

from sklearn.model_selection import train_test_split

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score

import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('../input/heart.csv')

df.head()
df.corr()['target']
sns.countplot(x='target',data=df)
table = pd.pivot_table(df, values='target',index='sex')

table
df.isna().sum()
df.dtypes
y = df.target

x = df.drop(labels='target', axis=1)
pd.crosstab(df.sex,y)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=17)
clf = LogisticRegressionCV(cv=16, random_state=17,max_iter=100, solver='liblinear')

clf.fit(x_train,y_train)

clf.predict(x_test)

clf.predict_proba(x_test).shape

lrcv_score = clf.score(x_test,y_test)

lrcv_score
params = {

        'min_child_weight': [1, 5, 10, 15],

        'gamma': [0.5, 1, 1.5, 2, 5],

        'subsample': [0.2, 0.6, 0.8, 1.0],

        'colsample_bytree': [0.6, 0.8, 1.0],

        'max_depth': [3, 4, 5, 6],

        'learning_rate': [0.01, 0.03, 0.05],

        'n_estimators': [50, 100, 200, 400, 600]

}



folds = 20

param_comb = 30



xgb = XGBClassifier(learning_rate=0.01,n_estimators=300,objective='binary:logistic',silent=True,nthread=1)





skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 17)

random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring='roc_auc', cv=skf.split(x_train,y_train), verbose=0, random_state=17 )



random_search.fit(x_train,y_train)
predictions = random_search.predict(x_test)

from sklearn.metrics import mean_absolute_error

print(str(mean_absolute_error(predictions, y_test)))
my_model = XGBClassifier()

parameters = {

        'n_estimators': [50, 100, 200, 500, 1000],

        'max_depth': [3,5,7,9],

        'learning_rate': [0.01, 0.1]

}

grid_search = GridSearchCV(estimator=my_model,param_grid=parameters,cv=5)

grid_search.fit(x_train,y_train,verbose=False)

y_pred = grid_search.predict(x_test)

predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y_test, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))