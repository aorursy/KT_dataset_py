# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn import preprocessing

import matplotlib.pyplot as plt 

plt.rc("font", size=14)

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

import seaborn as sns

sns.set(style="white")

sns.set(style="whitegrid", color_codes=True)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/HR-Employee-Attrition.csv")

df.head()
df.info()
df.shape
df.duplicated().sum()
df.isna().sum()
import pandas_profiling

pandas_profiling.ProfileReport(df)


df.drop(columns=["StandardHours","MonthlyIncome","EmployeeCount","EmployeeNumber","DailyRate"],inplace=True)

df.head()
df.info()
df.Gender.unique()
df.Attrition.replace({"Yes":1,"No":0}, inplace=True)

df.OverTime.replace({"Yes":1,"No":0}, inplace=True)

df.Over18.replace({"Y":1,"N":0}, inplace=True)
df_column_numeric = df.select_dtypes(include=np.number).columns

df_column_category = df.select_dtypes(exclude=np.number).columns
print("df_column_numeric",df_column_numeric)

print("df_column_category",df_column_category)
df_category_onehot = pd.get_dummies(df[df_column_category])
df_final = pd.concat([df_category_onehot,df[df_column_numeric]], axis = 1)
df_final.corr()
df_final['Attrition'].value_counts()
%matplotlib inline

pd.crosstab(df.Department,df.Attrition).plot(kind='bar')

plt.title('Attrition rate based on Department')

plt.xlabel('Department')

plt.ylabel('Attrition')
%matplotlib inline

pd.crosstab(df.Gender,df.Attrition).plot(kind='bar')

plt.title('Attrition rate based on Gender')

plt.xlabel('Gender')

plt.ylabel('Attrition')
%matplotlib inline

pd.crosstab(df.YearsWithCurrManager,df.Attrition).plot(kind='bar')

plt.title('Attrition rate based on YearsWithCurrManager')

plt.xlabel('YearsWithCurrManager')

plt.ylabel('Attrition')
%matplotlib inline

pd.crosstab(df.JobLevel,df.Attrition).plot(kind='bar')

plt.title('Attrition rate based on YearsWithCurrManager')

plt.xlabel('YearsWithCurrManager')

plt.ylabel('Attrition')
x= df_final.drop(["Attrition"],axis=1)

x.head()
y=df["Attrition"]

y.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

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
mythreshold=0.7

test_Prob = logreg.predict_proba(X_test)

test_New_Pred = np.where(test_Prob[:,1] > mythreshold, 1, 0)

metrics.confusion_matrix(y_test,test_New_Pred)
metrics.accuracy_score(y_test,test_New_Pred)