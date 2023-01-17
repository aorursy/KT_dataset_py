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

dataf=pd.read_csv("../input/HR-Employee-Attrition.csv")
dataf.info()
dataf.head()
dataf.corr()
dataf['Attrition'].replace({'No':0,'Yes':1},inplace=True)
dataf.head()
encoded_BusinessTravel = pd.get_dummies(dataf["BusinessTravel"])



encoded_BusinessTravel.columns = ['BusinessTravel-' + str(col) for col in encoded_BusinessTravel.columns]



dataf=pd.concat([dataf, encoded_BusinessTravel], axis=1)
dataf.head(2)
encoded_Department = pd.get_dummies(dataf["Department"])



encoded_Department.columns = ['Department-' + str(col) for col in encoded_Department.columns]



dataf=pd.concat([dataf, encoded_Department], axis=1)
encoded_EducationField = pd.get_dummies(dataf["EducationField"])



encoded_EducationField.columns = ['EducationField-' + str(col) for col in encoded_EducationField.columns]



dataf=pd.concat([dataf, encoded_EducationField], axis=1)
encoded_Gender = pd.get_dummies(dataf["Gender"])



encoded_Gender.columns = ['Gender-' + str(col) for col in encoded_Gender.columns]



dataf=pd.concat([dataf, encoded_Gender], axis=1)
encoded_JobRole = pd.get_dummies(dataf["JobRole"])



encoded_JobRole.columns = ['JobRole-' + str(col) for col in encoded_JobRole.columns]



dataf=pd.concat([dataf, encoded_JobRole], axis=1)
encoded_MaritalStatus = pd.get_dummies(dataf["MaritalStatus"])



encoded_MaritalStatus.columns = ['MaritalStatus-' + str(col) for col in encoded_MaritalStatus.columns]



dataf=pd.concat([dataf, encoded_MaritalStatus], axis=1)
encoded_Over18 = pd.get_dummies(dataf["Over18"])



encoded_Over18.columns = ['Over18-' + str(col) for col in encoded_Over18.columns]



dataf=pd.concat([dataf, encoded_Over18], axis=1)
encoded_OverTime = pd.get_dummies(dataf["OverTime"])



encoded_OverTime.columns = ['OverTime-' + str(col) for col in encoded_OverTime.columns]



dataf=pd.concat([dataf, encoded_OverTime], axis=1)
import numpy as np

num_col = dataf.select_dtypes(include=np.number).columns
dataf.info()
obj_col = dataf.select_dtypes(exclude=np.number).columns
obj_col
X=dataf[num_col]

X = X.drop(columns='Attrition')

Y=dataf['Attrition']
X.head(2)
X=X.drop(columns=["StandardHours","EmployeeCount","Over18-Y"])
X.corr()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

logreg = LogisticRegression()

logreg.fit(X_train, y_train)
train_Pred = logreg.predict(X_train)
metrics.confusion_matrix(y_train,train_Pred)
metrics.accuracy_score(y_train,train_Pred)
test_Pred = logreg.predict(X_test)
metrics.confusion_matrix(y_test,test_Pred)
metrics.accuracy_score(y_test,test_Pred)
from sklearn.metrics import classification_report

print(classification_report(y_test, test_Pred))

"""The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives. The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.



The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives. The recall is intuitively the ability of the classifier to find all the positive samples.



The support is the number of occurrences of each class in y_test."""
from sklearn import preprocessing

import matplotlib.pyplot as plt 

plt.rc("font", size=14)

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

import seaborn as sns

sns.set(style="white")

sns.set(style="whitegrid", color_codes=True)
from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))

fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])

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