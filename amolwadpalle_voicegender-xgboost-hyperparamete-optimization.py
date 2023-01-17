# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn.model_selection import train_test_split

import numpy as np

import xgboost as xgb

from xgboost.sklearn import XGBClassifier

from sklearn.metrics import accuracy_score
df = pd.read_csv('/kaggle/input/voicegender/voice.csv')

df['label'].replace('male', 1, inplace = True)

df['label'].replace('female', 0, inplace = True)

X = np.array(df.drop(['label'], axis = 1))

Y = df['label']

# mylabel = LabelEncoder()

# y = mylabel.fit_transform(Y)





X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size = 0.8, test_size = 0.2, random_state = 42)



print('X-Train Size : ', X_train.shape)

print('Y-Train Size : ', y_train.shape)

print('X-test Size : ', X_test.shape)

print('y-test Size : ', y_test.shape)
def modelfit(clf, X_train, y_train, X_test, y_test):

    clf.fit(X_train, y_train)

    predection = clf.predict(X_test)

    accuracy = accuracy_score(y_test, predection)

    print(accuracy)
base_clf = XGBClassifier()

modelfit(base_clf, X_train, y_train, X_test, y_test)
from sklearn.model_selection import RandomizedSearchCV

clf = XGBClassifier()



n_estimators =range(100,1000, 100)

max_depth = range(3,10,2),

min_child_weight = range(1,6,2)

param_grid = {'n_estimators': n_estimators,

              'learning_rate' : [0.025, 0.05, 0.1],

              'max_depth' : max_depth,

              'min_child_weight' : min_child_weight,

              'max_depth':[4,5,6],

              'min_child_weight':[6,8,10,12],

              'gamma':[i/10.0 for i in range(0,5)]}



random_grid = RandomizedSearchCV(estimator = clf, param_distributions = param_grid, n_jobs= -1, verbose=0)

print(random_grid)



random_grid.fit(X_train, y_train)
random_grid.best_params_
from sklearn.model_selection import GridSearchCV



clf = XGBClassifier()

param_grid = {'n_estimators': [60,100,160, 200, 230],

              'learning_rate': [0.025, 0.05],

              'min_child_weight': [7,8,9], 

              'max_depth': [3,4,5,6,7], 

              'gamma': [0.3, 0.4, 0.2]

             }

grid_search = GridSearchCV(estimator = clf, param_grid = param_grid, n_jobs=-1)



grid_search.fit(X_train, y_train)

grid_search.best_params_
best_model = modelfit(grid_search.best_estimator_, X_train, y_train, X_test, y_test)
from sklearn.model_selection import cross_val_score

cfl = XGBClassifier()



K_score = cross_val_score(cfl, X, Y, cv=10, n_jobs=-1)

K_score
K_score.mean()
import numpy as np  # linear algebra

import pandas as pd  # read and wrangle dataframes

import matplotlib.pyplot as plt # visualization

import seaborn as sns # statistical visualizations and aesthetics

from sklearn.base import TransformerMixin # To create new classes for transformations

from sklearn.preprocessing import (FunctionTransformer, StandardScaler) # preprocessing 

from sklearn.decomposition import PCA # dimensionality reduction

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from scipy.stats import boxcox # data transform

from sklearn.model_selection import (train_test_split, KFold , StratifiedKFold, 

                                     cross_val_score, GridSearchCV, 

                                     learning_curve, validation_curve) # model selection modules

from sklearn.pipeline import Pipeline # streaming pipelines

from sklearn.base import BaseEstimator, TransformerMixin # To create a box-cox transformation class

from collections import Counter

import warnings

# load models

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from xgboost import (XGBClassifier, plot_importance)

from sklearn.svm import SVC

from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier)

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from time import time



%matplotlib inline 

warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
model_importances = XGBClassifier()

start = time()

model_importances.fit(X_train, y_train)

print('Elapsed time to train XGBoost  %.3f seconds' %(time()-start))

plot_importance(model_importances)

plt.show()