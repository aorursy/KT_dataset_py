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

import seaborn as sns # data visualisation

import matplotlib.pyplot as plt # plot

from statsmodels.stats.outliers_influence import variance_inflation_factor

credit= pd.read_csv("../input/creditcard.csv")

credit.head()
# ad=credit.corr()

# ad.style.background_gradient(cmap='coolwarm')
# from statsmodels.stats.outliers_influence import variance_inflation_factor



# train1 = credit._get_numeric_data() #drop non-numeric cols

# # Y=train.status_group

# # X=train1.drop(['status_group'], axis=1)

# vif = pd.DataFrame()

# vif["VIF_Factor"] = [variance_inflation_factor(train1.values, i) for i in range(train1.shape[1])]

# vif["features"] = train1.columns

# pd.DataFrame(list(sorted(zip(vif.VIF_Factor, vif.features))))
credit.skew().sort_values(ascending=False)[:10]
sns.distplot(credit['Amount'])
credit.Amount=np.log1p(credit['Amount'])

sns.distplot(credit.Amount)
credit.Amount.skew()
from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split

#from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

# demonstration of calculating metrics for a neural network model using sklearn

from sklearn.datasets import make_circles

from sklearn.metrics import cohen_kappa_score

from sklearn.metrics import roc_auc_score

from imblearn.over_sampling import SMOTE

from imblearn.under_sampling import NearMiss

from imblearn.over_sampling import ADASYN

from sklearn.metrics import cohen_kappa_score

from sklearn.metrics import average_precision_score, auc, roc_curve, precision_recall_curve
credit=credit.drop(['V23','V24','V25','V22','V13','V14','V1','V5','V15','V19','V20','V8','V28','V27','Time','V21','V6','V2'], axis=1)
Y=credit.Class

X=credit.drop(['Class'], axis=1)

#X = pd.get_dummies(X)
from sklearn.feature_selection import RFE 

from sklearn.svm import SVR 
from sklearn.feature_selection import RFE 

from sklearn.svm import SVR 

Y=credit.Class

X=credit.drop(['Class'], axis=1)

estimator = SVR(kernel="linear") 

selector = RFE(estimator,10,step=1) 

selector = selector.fit(X, Y)

selector.ranking_

list(zip(X.columns,selector.ranking_))

#Splitting into test and train

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3,random_state=32, stratify= Y)
from sklearn.tree import DecisionTreeClassifier 

model = DecisionTreeClassifier(criterion='gini', max_depth=10,min_samples_split=10 ,random_state=99) 

model.fit(X_train, y_train)



x_predrf = model.predict(X_train)

y_predrf = model.predict(X_test)



print('Classification_Report',classification_report(y_test,y_predrf ))

print('KAPPA SCORE: ',cohen_kappa_score(y_test,y_predrf ))

print('AUC: ', roc_auc_score(y_test,y_predrf))

print('Avg Precesion Score', average_precision_score(y_test,y_predrf))

pd.crosstab(y_test, y_predrf, rownames=['True'], colnames=['Predicted'], margins=True)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components=5)

lda.fit(X_train, y_train)

x_predrf = lda.predict(X_train)

y_predrf = lda.predict(X_test)



print('Classification_Report',classification_report(y_test,y_predrf ))

print('KAPPA SCORE: ',cohen_kappa_score(y_test,y_predrf))

print('AUC: ', roc_auc_score(y_test,y_predrf))

print('Avg Precesion Score', average_precision_score(y_test,y_predrf))

pd.crosstab(y_test, y_predrf, rownames=['True'], colnames=['Predicted'], margins=True)
from collections import Counter

from imblearn.combine import SMOTETomek

sm = SMOTETomek(ratio=0.3)

X_res, y_res = sm.fit_sample(X, Y)

print('Resampled dataset shape %s' % Counter(y_res))

print(X_res.shape, y_res.shape)
# after feature selection

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components=5)

lda.fit(X_res, y_res)

x_predrf = lda.predict(X_res)

y_predrf = lda.predict(X_test)



# print('Confusion_Matrix',confusion_matrix(y_test, y_predrf))

print('Classification_Report',classification_report(y_test,y_predrf ))

print('KAPPA SCORE: ',cohen_kappa_score(y_test,y_predrf))

print('AUC: ', roc_auc_score(y_test,y_predrf))

print('Avg Precesion Score', average_precision_score(y_test,y_predrf))

pd.crosstab(y_test, y_predrf, rownames=['True'], colnames=['Predicted'], margins=True)
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state=222, max_depth= 8 )

classifier.fit(X_train, y_train)

x_predrf = classifier.predict(X_train)

y_predrf = classifier.predict(X_test)



print('Confusion_Matrix',confusion_matrix(y_test, y_predrf))

print('Classification_Report',classification_report(y_test,y_predrf ))

print('KAPPA SCORE: ',cohen_kappa_score(y_test,y_predrf))

print('AUC: ', roc_auc_score(y_test,y_predrf))

print('Avg Precesion Score', average_precision_score(y_test,y_predrf))
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state=22, max_depth= 10, class_weight={0:1,1:3})

classifier.fit(X_train, y_train)

x_predrf = classifier.predict(X_train)

y_predrf = classifier.predict(X_test)



print('Confusion_Matrix',confusion_matrix(y_test, y_predrf))

print('Classification_Report',classification_report(y_test,y_predrf ))



print('KAPPA SCORE: ',cohen_kappa_score(y_test,y_predrf))

print('AUC: ', roc_auc_score(y_test,y_predrf))

print('Avg Precesion Score', average_precision_score(y_test,y_predrf))
