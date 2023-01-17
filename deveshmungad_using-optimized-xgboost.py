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

import numpy as np

import matplotlib.pyplot as plt

import xgboost as xgb

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



df=pd.read_csv(r"../input/titanic/train.csv")

df.info()

print(df.head())

df['Survived'].plot(kind="hist")
n_bins=10

dfs=df[df["Survived"]==1]

dfns=df[df["Survived"]==0]

x=dfs["Age"], dfns["Age"]

plt.hist(x, n_bins, histtype="bar", stacked=True)
df["Sexn"]=np.nan

df["Embarkedn"]=np.nan

for i in df.index:

    if(df["Sex"][i]=="male"):

        df["Sexn"][i]=1

    else:

        df["Sexn"][i]=2

    if(df["Embarked"][i]=="C"):

        df["Embarkedn"][i]=1

    elif(df["Embarked"][i]=="S"):

        df["Embarkedn"][i]=2

    else:

        df["Embarkedn"][i]=3



print(df.describe())
dfy=df[["Survived"]]

y=np.array(dfy)

dfx=df.drop(columns =['PassengerId', 'Survived','Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'])

X=np.array(dfx)



X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.1, random_state=10)
xgbt=xgb.XGBClassifier()

xgbt.fit(X, y)

xgby_pred=xgbt.predict(X_test)

xgb_score=accuracy_score(y_test, xgby_pred)

print(xgb_score) #normal
pg = {  'eta': [0.05, 0.1, 0.2, 0.3],

        'min_child_weight': [1, 5, 10],

        'gamma': [0.5, 1, 1.5, 2, 5],

        'subsample': [0.6, 0.8, 1.0],

        'colsample_bytree': [0.6, 0.8, 1.0],

        'max_depth': [4, 5, 6, 7, 8, 9, 10]

        }

gs=GridSearchCV(estimator=xgbt, param_grid=pg, scoring='accuracy')

gs.fit(X, y.ravel())

print(gs.best_estimator_)

print(gs.best_params_)

print(gs.best_score_)
xgbgs=xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,

              colsample_bynode=1, colsample_bytree=0.8, eta=0.3, gamma=2,

              gpu_id=-1, importance_type='gain', interaction_constraints='',

              learning_rate=0.300000012, max_delta_step=0, max_depth=9,

              min_child_weight=1, missing=np.nan, monotone_constraints='()',

              n_estimators=100, n_jobs=0, num_parallel_tree=1, random_state=0,

              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1.0,

              tree_method='exact', validate_parameters=1, verbosity=None)

xgbgs.fit(X, y)

xgbgsy_pred=xgbgs.predict(X)

xgbgs_score=accuracy_score(y, xgbgsy_pred)

print(xgbgs_score)
from scipy import stats



pr = {  'eta': [0.01, 0.05, 0.1, 0.2, 0.3, 0.4],

        'min_child_weight': stats.randint(1,10),

        'gamma': stats.uniform(0.5,5),

        'subsample': stats.uniform(0.6,1),

        'colsample_bytree': stats.uniform(0.6,1),

        'max_depth': stats.randint(4,11)

        }

rs=RandomizedSearchCV(estimator=xgbt, param_distributions=pr, scoring='accuracy', n_iter=5000)

rs.fit(X, y)

print(rs.best_estimator_)

print(rs.best_params_)

print(rs.best_score_)
xgbrs=xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,

                        colsample_bynode=1, colsample_bytree=0.7446784150274387, eta=0.4,

                        gamma=3.8796527603109134, gpu_id=-1, importance_type='gain',

                        interaction_constraints='', learning_rate=0.300000012,

                        max_delta_step=0, max_depth=10, min_child_weight=1, missing=np.nan,

                        monotone_constraints='()', n_estimators=100, n_jobs=0,

                        num_parallel_tree=1, random_state=0, reg_alpha=0, reg_lambda=1,

                        scale_pos_weight=1, subsample=0.6690824474441066,

                        tree_method='exact', validate_parameters=1, verbosity=None)

xgbrs.fit(X, y)

xgbrsy_pred=xgbrs.predict(X_test)

xgbrs_score=accuracy_score(y_test, xgbrsy_pred)

print(xgbrs_score)
dftest=pd.read_csv(r"../input/titanic/test.csv")



dftest["Sexn"]=np.nan

dftest["Embarkedn"]=np.nan

for i in dftest.index:

    if(dftest["Sex"][i]=="male"):

        dftest["Sexn"][i]=1

    else:

        dftest["Sexn"][i]=2

    if(dftest["Embarked"][i]=="C"):

        dftest["Embarkedn"][i]=1

    elif(dftest["Embarked"][i]=="S"):

        dftest["Embarkedn"][i]=2

    else:

        dftest["Embarkedn"][i]=3



dfpredx=dftest.drop(columns =['PassengerId', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'])

Xpred=np.array(dfpredx)

print(Xpred)
y_pred = xgbgs.predict(Xpred)

dfsubmission=dftest

dfsubmission["Survived"]=y_pred

dfsubmission=dfsubmission[["PassengerId", "Survived"]]

dfsubmission.set_index("PassengerId", inplace=True)

print(dfsubmission.head(20))



dfsubmission.to_csv("./submission.csv")