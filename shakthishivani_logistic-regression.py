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

import matplotlib.pyplot as  plt

%matplotlib inline

import seaborn as sns

employee_data = pd.read_csv("../input/HR-Employee-Attrition.csv")
employee_data.shape
employee_data.head()
employee_data.info()
employee_data.isna().sum()
employee_data.duplicated().sum()
num_cols = employee_data.select_dtypes(include=np.number).columns

cat_cols = employee_data.select_dtypes(exclude=np.number).columns
print(num_cols)

print(cat_cols)
employee_data[cat_cols].apply(lambda x:print(x.value_counts()))
employee_data.Attrition.replace({"Yes":1,"No":0},inplace = True)
employee_data.Over18.replace({"Y":1},inplace = True)
employee_data.OverTime.replace({"Yes":1,"No":0},inplace = True)
employee_data.info()
employee_data_onehot = pd.get_dummies(employee_data[cat_cols.drop(["Attrition","Over18","OverTime"])])
employee_final = pd.concat([employee_data_onehot,employee_data[num_cols],employee_data["Attrition"],employee_data["Over18"],employee_data["OverTime"]], axis = 1)
employee_final.head()
employee_final.shape
employee_final.corr()
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn import metrics
X=employee_final.drop(columns=['Attrition'])

y=employee_final[['Attrition']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

logreg = LogisticRegression()

logreg.fit(X_train, y_train)
train_pred = logreg.predict(X_train)
metrics.confusion_matrix(y_train,train_pred)
metrics.accuracy_score(y_train,train_pred)
test_pred = logreg.predict(X_test)
metrics.confusion_matrix(y_test,test_pred)
metrics.accuracy_score(y_test,test_pred)
from sklearn.metrics import classification_report

print(classification_report(y_test, test_pred))
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