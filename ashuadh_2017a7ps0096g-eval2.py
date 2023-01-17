import pandas as pd

import numpy as np

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import AdaBoostRegressor

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import VotingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neighbors import KNeighborsRegressor

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC

from sklearn.svm import SVR

from sklearn.naive_bayes import GaussianNB

import xgboost as xgb

import matplotlib.pyplot as plt

import lightgbm as lgb

from sklearn.model_selection import GridSearchCV

%matplotlib inline

df = pd.read_csv('/kaggle/input/eval-lab-2-f464/train.csv')

test = pd.read_csv('/kaggle/input/eval-lab-2-f464/test.csv')

df_submit = pd.DataFrame()

df_submit['id'] = test['id']

df.drop(columns = ['id'], inplace = True)

test.drop(columns = ['id'], inplace = True)
df.head()
df.describe()
import seaborn as sns

sns.countplot(x = df['class'], data =df)
df.corr()
#high correlations therefore we can drop some columns

#keep columns with high correlation with class

df.drop(columns = ['chem_0', 'chem_2', 'chem_3', 'chem_7'], inplace = True)

test.drop(columns = ['chem_0', 'chem_2', 'chem_3', 'chem_7'], inplace = True)
rfc = RandomForestClassifier()

bc = BaggingClassifier()

etc = ExtraTreesClassifier()

xgc = xgb.XGBClassifier()

lgc = lgb.LGBMClassifier()

clfs = [rfc, etc, bc, xgc, lgc]
param_grid = {

    'n_estimators': [100, 200, 500, 1000, 2000, 5000]

}
X = df.drop(columns = ['class'])

y = df['class']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state =  42)
for clf in clfs:

    grid = GridSearchCV(clf, param_grid = param_grid, cv = 5, verbose = 1)

    grid.fit(X_train, y_train)

    print(grid.best_params_)
rfc = RandomForestClassifier(n_estimators = 2000)

bc = BaggingClassifier(n_estimators = 1000)

etc = ExtraTreesClassifier(n_estimators = 100)

xgc = xgb.XGBClassifier(n_estimators = 5000, max_depth=3, learning_rate = 0.1)

lgc = lgb.LGBMClassifier(n_estimators = 500)
clfs = [rfc, bc, etc, xgc, lgc]

for clf in clfs:

    clf.fit(X_train, y_train)

    print(clf.score(X_test, y_test))
rfc.fit(X, y)

df_submit['class'] = rfc.predict(test)

df_submit.to_csv('rfc.csv', index = False)
vc = VotingClassifier(estimators=[('rfc', rfc), ('etc', etc), ('xgc', xgc), ('bc', bc), ('lgc', lgc)], weights=[4, 4, 1, 1, 1])
vc.fit(X_train, y_train)
vc.score(X_test, y_test)
vc.fit(X, y)

df_submit['class'] = vc.predict(test)

df_submit.to_csv('vc2.csv', index = False)