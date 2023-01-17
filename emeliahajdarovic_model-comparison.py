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
df=pd.read_csv("../input/data-ready-for-model/csvfile.csv")

df.head()
orig_df=pd.read_csv("../input/default-of-credit-card-clients-dataset/UCI_Credit_Card.csv")

orig_df.head()
target=df['default.payment.next.month']
df_=df[df.columns[~df.columns.isin(['default.payment.next.month'])]]#already does not have ID

orig_df_=orig_df[orig_df.columns[~orig_df.columns.isin(['default.payment.next.month','ID'])]]
df_.head()
orig_df_.head()
from sklearn.model_selection import train_test_split, cross_val_score



# create X (data features) and y (target)

X = df_

target=df['default.payment.next.month']

y = target



# use train/test split with different random_state values

# we can change the random_state values that changes the accuracy scores

# the scores change a lot, this is why testing scores is a high-variance estimate

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 1)





sets=[X_train,X_test,y_train,y_test]



for s in sets:

    print(s.shape)
from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import f1_score

from sklearn.metrics import classification_report

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

from catboost import CatBoostClassifier

from sklearn.metrics import roc_auc_score





## List of ML Algorithms, we will use for loop for each algorithms.

models = [LogisticRegression(solver = "liblinear"),

          DecisionTreeClassifier(),

          RandomForestClassifier(n_estimators =10),

          XGBClassifier(),

          GradientBoostingClassifier(),

          LGBMClassifier(),GaussianNB()

         ]





for model in models:

    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)

    proba = model.predict_proba(X_test)

    roc_score = roc_auc_score(y_test, proba[:,1])

    cv_score = cross_val_score(model,X_train,y_train,cv=10).mean()

    score = accuracy_score(y_test,y_pred)

    bin_clf_rep = classification_report(y_test,y_pred, zero_division=1)

    name = str(model)

    print(name[0:name.find("(")])

    print("Accuracy :", score)

    print("CV Score :", cv_score)

    print("AUC Score : ", roc_score)

    print(bin_clf_rep)

    print(confusion_matrix(y_test,y_pred))

    print("------------------------------------------------------------")
from sklearn.model_selection import train_test_split, cross_val_score



# create X (data features) and y (target)

X = orig_df_

target=df['default.payment.next.month']

y = target



# use train/test split with different random_state values

# we can change the random_state values that changes the accuracy scores

# the scores change a lot, this is why testing scores is a high-variance estimate

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 1)





sets=[X_train,X_test,y_train,y_test]



for s in sets:

    print(s.shape)
from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import f1_score

from sklearn.metrics import classification_report

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

from catboost import CatBoostClassifier

from sklearn.metrics import roc_auc_score





## List of ML Algorithms, we will use for loop for each algorithms.

models = [LogisticRegression(solver = "liblinear"),

          DecisionTreeClassifier(),

          RandomForestClassifier(n_estimators =10),

          XGBClassifier(),

          GradientBoostingClassifier(),

          LGBMClassifier(),GaussianNB()

         ]





for model in models:

    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)

    proba = model.predict_proba(X_test)

    roc_score = roc_auc_score(y_test, proba[:,1])

    cv_score = cross_val_score(model,X_train,y_train,cv=10).mean()

    score = accuracy_score(y_test,y_pred)

    bin_clf_rep = classification_report(y_test,y_pred, zero_division=1)

    name = str(model)

    print(name[0:name.find("(")])

    print("Accuracy :", score)

    print("CV Score :", cv_score)

    print("AUC Score : ", roc_score)

    print(bin_clf_rep)

    print(confusion_matrix(y_test,y_pred))

    print("------------------------------------------------------------")