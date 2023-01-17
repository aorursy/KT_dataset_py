# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as  plt

%matplotlib inline

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/HR-Employee-Attrition.csv")
df.head()
df.info()
df.isna().sum()
df.notnull().sum()
df.describe()
df[['EmployeeCount','EmployeeNumber']]

df.head()
df['StandardHours'].value_counts()

df['StandardHours'].unique()
df.drop(['EmployeeCount','EmployeeNumber','Over18','StandardHours'], axis = 1, inplace = True)
df.head()
df.dtypes
numericalColumns = df.select_dtypes(include=np.number).columns

categoricalColumns = df.select_dtypes(exclude=np.number).columns
print(numericalColumns)

print(categoricalColumns)
ls = list(df[categoricalColumns])

for i in ls:

    print(i,":\n",df[i].unique(),'value Counts :',df[i].value_counts())

    print('-----------------------------------------------------------')
df.Attrition.replace({"Yes":1,"No":0},inplace = True)
df.head()
# df.Attrition.replace({"Yes":1,"No":0},inplace=True)

# df.Gender.replace({"Male":1,"Female":0},inplace=True)

# df.MaritalStatus.replace({"Single":1,"Married":0},inplace=True)
encodedCatCol = pd.get_dummies(df[categoricalColumns.drop(["Attrition"])])

encodedCatCol.head()
df_encoded_onehot = pd.concat([df[numericalColumns],encodedCatCol], axis = 1)

df_encoded_onehot.head()
df_final =pd.concat([df_encoded_onehot,df["Attrition"]],axis=1)
df_final.head()
df_final.corr()
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split 
X = df_final.drop(columns='Attrition')

Y = df_final[['Attrition']]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=0)

logreg = LogisticRegression()

logreg.fit(X_train,Y_train)
train_pred = logreg.predict(X_train)
from sklearn import metrics
metrics.confusion_matrix(Y_train,train_pred)
tn, fp, fn, tp = metrics.confusion_matrix(Y_train,train_pred).ravel()

specificity = tn / (tn+fp)



print(specificity)
metrics.accuracy_score(Y_train,train_pred)
test_pred = logreg.predict(X_test)
metrics.confusion_matrix(Y_test, test_pred)
metrics.accuracy_score(Y_test, test_pred)
from sklearn.metrics import classification_report

print(classification_report(Y_test,test_pred))
from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

ROC_auc = roc_auc_score(Y_test,test_pred)

fpr, tpr, thresholds = roc_curve(Y_test,logreg.predict_proba(X_test)[:,1]) 

plt.figure()

plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % ROC_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.savefig('Log_ROC')

plt.show()



print(ROC_auc)

#print(thresholds)

#print(fpr)