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

import matplotlib.pyplot as plt

import seaborn as sns

import sklearn as sk

from sklearn.feature_selection import RFE

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

import statsmodels.api as sm

from sklearn import metrics

from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve 
train_df = pd.read_csv('../input/train.csv')
X_prestd = train_df.iloc[:,2:]

X = (X_prestd - X_prestd.mean())/(X_prestd.std())

Y = train_df.iloc[:,1]
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.2, random_state = 42)
logreg = LogisticRegression()

rfe = RFE(logreg, 50)

rfe = rfe.fit(X_train, Y_train)
X_train1 = X_train.loc[:,X_train.columns[rfe.support_]]

logit_model = sm.Logit(Y_train, X_train1)

results = logit_model.fit() 
logreg = LogisticRegression()

logreg.fit(X_train1, Y_train)

X_val1 = X_val.loc[:,X_val.columns[rfe.support_]]

Y_pred_val = logreg.predict(X_val1)
confusion_matrix = confusion_matrix(Y_val, Y_pred_val)
logit_roc_auc = roc_auc_score(Y_val, logreg.predict(X_val1))

fpr, tpr, thresholds = roc_curve(Y_val, logreg.predict_proba(X_val1)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.savefig('Log_ROC')

plt.show() 

test_df = pd.read_csv('../input/test.csv') 

X_test_prestd = test_df.iloc[:,1:]

X_test_full = (X_test_prestd - X_test_prestd.mean())/(X_test_prestd.std())

X_test = X_test_full.loc[:,X_test_full.columns[rfe.support_]]

Y_pred = logreg.predict(X_test)
Result_df = pd.DataFrame(Y_pred)
test_x = test_df.iloc[:,0]

Field_df = pd.DataFrame(test_x)
Final = pd.merge(Field_df, Result_df, left_index=True, right_index=True)
Final.to_csv("submission.csv")