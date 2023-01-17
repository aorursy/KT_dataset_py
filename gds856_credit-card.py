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
# Loading the libraries



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import pandas_profiling

pd.set_option('display.max_columns', 1000)

pd.options.display.float_format = '{:.4f}'.format



from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

from sklearn.metrics import cohen_kappa_score

from sklearn.metrics import roc_auc_score

from imblearn.over_sampling import SMOTE, BorderlineSMOTE

from sklearn.metrics import average_precision_score, auc, roc_curve, precision_recall_curve



from xgboost import XGBClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.ensemble import RandomForestClassifier

from collections import Counter

from imblearn.combine import SMOTETomek, SMOTEENN

from imblearn.over_sampling import BorderlineSMOTE
# Reading the I/P file



credit=pd.read_csv("../input/creditcard.csv")

credit.head(6)
credit.describe()
# Checking the Profile Report



pandas_profiling.ProfileReport(credit)
# Removing Duplicate rows

credit.drop_duplicates(keep=False, inplace=True)
credit.shape
# checking the fraud and non-fraud data

fraud=credit.loc[credit['Class'] == 1]

notfraud=credit.loc[credit['Class'] == 0]
len(fraud.loc[fraud['Amount'] == 0.00])
len(notfraud.loc[notfraud['Amount'] == 0.00])
# Checking Skewness of data

credit.skew()
Y = credit.Class

X = credit.drop(['Class'], axis=1)
#Splitting into test and train



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=32, stratify= Y)
from sklearn.tree import DecisionTreeClassifier 

model = DecisionTreeClassifier(criterion='gini', max_depth=10,min_samples_split=10 ,random_state=99) 

model.fit(X_train, y_train)



x_predrf = model.predict(X_train)

y_predrf = model.predict(X_test)



print('Confusion_Matrix\n',confusion_matrix(y_test, y_predrf))

print('Classification_Report\n',classification_report(y_test,y_predrf ))

print('AUC: ', roc_auc_score(y_test,y_predrf))

print('Avg Precesion Score', average_precision_score(y_test,y_predrf))

print('KAPPA SCORE: ',cohen_kappa_score(y_test,y_predrf ))



pd.crosstab(y_test, y_predrf, rownames=['True'], colnames=['Predicted'], margins=True)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components=5)

lda.fit(X_train, y_train)

x_predrf = lda.predict(X_train)

y_predrf = lda.predict(X_test)



print('Confusion_Matrix\n',confusion_matrix(y_test, y_predrf))

print('Classification_Report\n',classification_report(y_test,y_predrf ))

print('AUC: ', roc_auc_score(y_test,y_predrf))

print('Avg Precesion Score', average_precision_score(y_test,y_predrf))

print('KAPPA SCORE: ',cohen_kappa_score(y_test,y_predrf ))



pd.crosstab(y_test, y_predrf, rownames=['True'], colnames=['Predicted'], margins=True)
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 100, random_state=22, max_depth= 10)

classifier.fit(X_train, y_train)

x_predrf = classifier.predict(X_train)

y_predrf = classifier.predict(X_test)



print('Confusion_Matrix\n',confusion_matrix(y_test, y_predrf))

print('Classification_Report\n',classification_report(y_test,y_predrf ))

print('AUC: ', roc_auc_score(y_test,y_predrf))

print('Avg Precesion Score', average_precision_score(y_test,y_predrf))

print('KAPPA SCORE: ',cohen_kappa_score(y_test,y_predrf ))



pd.crosstab(y_test, y_predrf, rownames=['True'], colnames=['Predicted'], margins=True)
classifier = RandomForestClassifier(n_estimators = 50, criterion = 'entropy', random_state=22, max_depth= 10, class_weight={0:1,1:3})

classifier.fit(X_train, y_train)

x_predrf = classifier.predict(X_train)

y_predrf = classifier.predict(X_test)



print('Confusion_Matrix\n',confusion_matrix(y_test, y_predrf))

print('Classification_Report',classification_report(y_test,y_predrf ))

print('KAPPA SCORE: ',cohen_kappa_score(y_test,y_predrf))

print('AUC: ', roc_auc_score(y_test,y_predrf))

print('Avg Precesion Score', average_precision_score(y_test,y_predrf))

pd.crosstab(y_test, y_predrf, rownames=['True'], colnames=['Predicted'], margins=True)
from imblearn.combine import SMOTEENN



sm = SMOTEENN(ratio=0.1)

X_res, y_res = sm.fit_sample(X_train, y_train)

print('Resampled dataset shape %s' % Counter(y_res))

print(X_res.shape, y_res.shape)
# after OverSampling 0.1

classifier = RandomForestClassifier(n_estimators = 50, criterion = 'entropy', random_state=22, max_depth= 10, class_weight={0:1,1:3})

classifier.fit(X_res, y_res)

x_predrf = classifier.predict(X_res)

y_predrf = classifier.predict(X_test)



print('Confusion_Matrix\n',confusion_matrix(y_test, y_predrf))

print('Classification_Report\n',classification_report(y_test,y_predrf ))

print('AUC: ', roc_auc_score(y_test,y_predrf))

print('Avg Precesion Score', average_precision_score(y_test,y_predrf))

print('KAPPA SCORE: ',cohen_kappa_score(y_test,y_predrf ))

pd.crosstab(y_test, y_predrf, rownames=['True'], colnames=['Predicted'], margins=True)
from imblearn.over_sampling import BorderlineSMOTE



sm = BorderlineSMOTE()

X_res, y_res = sm.fit_sample(X, Y)

print('Resampled dataset shape %s' % Counter(y_res))

print(X_res.shape, y_res.shape)
# after OverSampling 

classifier = RandomForestClassifier(n_estimators = 50, criterion = 'entropy', random_state=22, max_depth= 10, class_weight={0:1,1:3})

classifier.fit(X_res, y_res)

x_predrf = classifier.predict(X_res)

y_predrf = classifier.predict(X_test)



print('Confusion_Matrix\n',confusion_matrix(y_test, y_predrf))

print('Classification_Report\n',classification_report(y_test,y_predrf ))

print('AUC: ', roc_auc_score(y_test,y_predrf))

print('Avg Precesion Score', average_precision_score(y_test,y_predrf))

print('KAPPA SCORE: ',cohen_kappa_score(y_test,y_predrf ))

pd.crosstab(y_test, y_predrf, rownames=['True'], colnames=['Predicted'], margins=True)