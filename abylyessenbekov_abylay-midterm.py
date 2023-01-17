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
train = pd.read_csv("/kaggle/input/santander-customer-transaction-prediction/train.csv")

test = pd.read_csv("/kaggle/input/santander-customer-transaction-prediction/test.csv")
train.isna().sum().sum()
# from sklearn.model_selection import train_test_split

# from sklearn.ensemble import RandomForestClassifier

# from sklearn.naive_bayes import GaussianNB

# from sklearn.tree import DecisionTreeClassifier

# from sklearn.linear_model import LogisticRegression

# from sklearn.model_selection import train_test_split

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split



from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.metrics import precision_recall_curve

from sklearn.metrics import average_precision_score

from sklearn.metrics import roc_curve

from sklearn.metrics import auc

from sklearn.model_selection import cross_val_score

from sklearn.metrics import f1_score

from sklearn.metrics import confusion_matrix

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score

y = train['target']

X = train.drop(columns = ['target', 'ID_code'])

test_v = test.drop(columns = ['ID_code'])
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state=0)

logreg = LogisticRegression(C=1)

logreg.fit(x_train, y_train)
prediction_y = logreg.predict(x_test)

roc_auc_score(y_test, prediction_y)
# NB

nbg = GaussianNB()

nbg.fit(x_train, y_train)
roc_auc_score(y_test, nbg.predict(x_test))
print("Accuracy Score is ",accuracy_score(y_test, nbg.predict(x_test)))
print(classification_report(y_test,nbg.predict(x_test)

                           ))
submission = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/sample_submission.csv', index_col='ID_code')

submission.to_csv("submission.csv")
# Decision Tree

dt = DecisionTreeClassifier()

dt.fit(x_train, y_train)
print('Accuracy score is :')

accuracy_score(y_test, dt.predict(x_test))
print('AUC_ROC_SCORE is :')

roc_auc_score(y_test, dt.predict(x_test))
# Random Forest

forest_model = RandomForestClassifier()

forest_model.fit(x_train, y_train)

melb_preds = forest_model.predict(x_test)

print(accuracy_score(y_test, melb_preds))
print('AUC_ROC_SCORE is :')

roc_auc_score(y_test, forest_model.predict(x_test))
from xgboost import XGBClassifier

xgb = XGBClassifier(max_depth=8,random_state=0)
xgb.fit(x_train, y_train)

xgb_pred = xgb.predict_proba(x_test)[:, 1]

performance(y_test, xgb_pred)