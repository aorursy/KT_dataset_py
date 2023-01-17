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
import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, cross_validate

from sklearn.metrics import roc_auc_score

from sklearn.metrics import mean_squared_error, r2_score, make_scorer



import xgboost as xgb

from sklearn.svm import SVC

import lightgbm as lgb

from sklearn.neural_network import MLPClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.ensemble import GradientBoostingClassifier

from lightgbm import LGBMRegressor

import optuna
target_col = "G3"
df_train = pd.read_csv("/kaggle/input/1056lab-student-performance-prediction/train.csv",index_col=0)

df_test = pd.read_csv("/kaggle/input/1056lab-student-performance-prediction/test.csv",index_col=0)
def root_mean_squared_error(y_true, y_pred):

    return np.sqrt(mean_squared_error(y_true, y_pred))
scoring = {"r2": "r2",

           "RMSE":make_scorer(root_mean_squared_error)}
df_train = df_train.replace({'GP':1,'MS':2,

                       'por':1,'mat':2,

                       'F':1,'M':2,

                       'U':1,'R':2,

                       'GT3':1,'LE3':2,

                       'T':1,'A':2,

                       'other':1,'services':2,'teacher':3,'health':4,'at_home':5,

                       'course':1,'reputation':2,'home':3,'other':4,

                       'mother':1,'father':2,'other':3,

                       True:1,False:2})

df_test = df_test.replace({'GP':1,'MS':2,

                       'por':1,'mat':2,

                       'F':1,'M':2,

                       'U':1,'R':2,

                       'GT3':1,'LE3':2,

                       'T':1,'A':2,

                       'other':1,'services':2,'teacher':3,'health':4,'at_home':5,

                       'course':1,'reputation':2,'home':3,'other':4,

                       'mother':1,'father':2,'other':3,

                       True:1,False:2})
X = df_train.drop('G3' ,axis=1).values

Xtest = df_test.values

y = df_train[target_col].values
clf = xgb.XGBRegressor()

clf.fit(X, y)

predict = clf.predict(Xtest)
submit = pd.read_csv('/kaggle/input/1056lab-student-performance-prediction/sampleSubmission.csv',index_col=0)

submit['G3'] = predict

submit.to_csv('submission.csv')