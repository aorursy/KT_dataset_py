# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import xgboost as xgb

from sklearn import preprocessing



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
test = pd.read_csv("/kaggle/input/santander-customer-transaction-prediction/test.csv") 

train = pd.read_csv("/kaggle/input/santander-customer-transaction-prediction/train.csv")
# train = train_large.drop(['ID_code', 'target'], axis = 1).astype('float32')

# train.info()
# train.head()
# train.isna().sum()
# train.nunique()
# train.dropna(inplace=True)
# X = train.iloc[:, 2:-1].values

# y = train.iloc[:, 1].values
# y[1]
# X[1]
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state = 1,stratify=y)
# from sklearn.neighbors import KNeighborsClassifier
# lab_enc = preprocessing.LabelEncoder()

# y_train_encoded = lab_enc.fit_transform(y_train)
# knn = KNeighborsClassifier(11)
# X_test[2:3]
# knn.fit(X_train, y_train)
# x_test =test.iloc[:,2:]
# y_preds = knn.predict(x_test)
from sklearn.linear_model import LogisticRegression

from sklearn import metrics
# logreg = LogisticRegression()

# logreg.fit(X_train, y_train)
# x_test =test.iloc[:,2:]
# y_preds = logreg.predict(x_test)

# print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
# submission = pd.DataFrame({

#         "ID_code": test["ID_code"],

#         "target": y_preds

#     })

# submission.to_csv('YerezhepK_LogReg.csv', index=False)
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# svclassifier = SVC()
# plt.scatter(X_train[:, 0], X_train[:, 4], c=y_train, cmap = 'spring')
# svclassifier.fit(X_train, y_train)
# x_test =test.iloc[:,2:]
# y_pred = svclassifier1.predict(X_test)
from sklearn.naive_bayes import GaussianNB
# gnb = GaussianNB()
# gnb.fit(X_train, y_train)
# x_test =test.iloc[:,2:]
# y_pred = gnb.predict(x_test)
# submission = pd.DataFrame({

#         "ID_code": test["ID_code"],

#         "target": y_pred

#     })

# submission.to_csv('YerezhepK_Naive_Gaussian.csv', index=False)
from sklearn import metrics
# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# from sklearn.metrics import classification_report, confusion_matrix

# print(confusion_matrix(y_test, y_pred))
# print("Accuracy:",metrics.classification_report(y_test, y_pred))
from sklearn.naive_bayes import BernoulliNB

from sklearn.model_selection import train_test_split
# bnb = BernoulliNB(binarize=0.0)
#  bnb.fit(X_train, y_train)

#  bnb.score(X_test, y_test)
# y_pred = bnb.predict(x_test)
# submission = pd.DataFrame({

#         "ID_code": test["ID_code"],

#         "target": y_pred

#     })

# submission.to_csv('YerezhepK_Naive_Bernoulli.csv', index=False)
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
# mnb = MultinomialNB(alpha=0.01)
# y_pred = mnb.predict(x_test)
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics 
from sklearn.tree import export_graphviz 
# clf = DecisionTreeClassifier()
# clf = clf.fit(X_train,y_train)
# x_test =test.iloc[:,2:]
# y_pred = clf.predict(x_test)
# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# submission = pd.DataFrame({

#         "ID_code": test["ID_code"],

#         "target": y_pred

#     })

# submission.to_csv('YerezhepK_DTRESS.csv', index=False)
# clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
# clf = clf.fit(X_train,y_train)
# x_test =test.iloc[:,2:]
# y_pred = clf.predict(x_test)
# submission = pd.DataFrame({

#         "ID_code": test["ID_code"],

#         "target": y_pred

#     })

# submission.to_csv('YerezhepK_DTRESS_entropy.csv', index=False)
from sklearn.ensemble import RandomForestRegressor
# regressor = RandomForestRegressor(n_estimators=20, random_state=0)

# regressor.fit(X_train, y_train)

# y_pred = regressor.predict(x_test)
# y_pred
# submission = pd.DataFrame({

#         "ID_code": test["ID_code"],

#         "target": y_pred

#     })

# submission.to_csv('YerezhepK_DTRESS_RandForReg.csv', index=False)
# import os

# os.environ['KMP_DUPLICATE_LIB_OK']='True'
import xgboost as xgb

import pandas as pd
# xg_cl = xgb.XGBClassifier(objective = 'binary:logistic', n_estimators = 10, seed=123 )
# xg_cl.fit(X_train, y_train)
# test_dataset = test
# x_test = test.iloc[:,2:]

# x_test
# test_dataset = test[:50000]

# test_dataset['ID_code']
# y_pred = xg_cl.predict(X_test)
# submission = pd.DataFrame({

#         "ID_code": test_dataset["ID_code"],

#         "target": y_pred

#     })

# submission.to_csv('YerezhepK_XGBOOST.csv', index=False)