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
df=pd.read_csv("../input/HR-Employee-Attrition.csv")

df.head(5)
df.shape
df.columns
df.info()
df.describe()
df.duplicated().sum()
df.isna().sum()
df["Attrition"].replace("Yes",0,inplace=True)
df["Attrition"].replace("No",1,inplace=True)
df["Attrition"].value_counts()
df['Attrition'].astype(str).astype(int)
df.corr()["Attrition"]
df.head(10)
cat_col = df.select_dtypes(exclude=np.number).columns
encoded_cat_col = pd.get_dummies(df[cat_col])
encoded_cat_col
num_col = df.select_dtypes(include=np.number).columns
emp_model = pd.concat([df[num_col],encoded_cat_col], axis = 1)
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, roc_curve, recall_score
X = emp_model.drop(columns="Attrition")

y = emp_model["Attrition"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

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
from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

import matplotlib.pyplot as plt 

plt.rc("font", size=14)

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