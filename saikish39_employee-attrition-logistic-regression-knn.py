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
df = pd.read_csv("../input/HR-Employee-Attrition.csv")
df.columns
df.info()
from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn import metrics

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')
df
df.corr().T
df.isna().sum()
df.duplicated().sum()
df["Attrition"].value_counts()
df.Attrition.replace({"Yes":1,"No":0}, inplace=True)
df['Attrition'].value_counts()
df['BusinessTravel'].value_counts()
df['Department'].value_counts()
df.drop(columns=['EmployeeCount','StandardHours'], inplace=True)

df.columns
df.corr()["Attrition"]
for col in df:

    pd.crosstab(df[col],df.Attrition).plot(kind='bar',color = ('black','red'),figsize=(10,5))
cat_col = df.select_dtypes(exclude=np.number).columns
num_col = df.select_dtypes(include=np.number).columns
encoded_cat_col = pd.get_dummies(df[cat_col])
encoded_cat_col.head(5)
final_model = pd.concat([df[num_col],encoded_cat_col], axis = 1)

final_model.head(5)
x = final_model.drop(columns="Attrition")

y = final_model["Attrition"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

logreg = LogisticRegression()

logreg.fit(x_train, y_train)

train_Pred = logreg.predict(x_train)
metrics.confusion_matrix(y_train,train_Pred)
Accuracy_Percent_Train = (metrics.accuracy_score(y_train,train_Pred))*100

Accuracy_Percent_Train
test_Pred = logreg.predict(x_test)
metrics.confusion_matrix(y_test,test_Pred)
Accuracy_Percent_Test = (metrics.accuracy_score(y_test,test_Pred))*100

Accuracy_Percent_Test
from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

plt.rc("font", size=18)

logit_roc_auc = roc_auc_score(y_test, logreg.predict(x_test))

false_positive, true_positive, thresholds = roc_curve(y_test, logreg.predict_proba(x_test)[:,1])

plt.figure()

plt.plot(false_positive, true_positive, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.savefig('Log_ROC')

plt.show()
from sklearn.preprocessing import Imputer

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
train_y = y_train.ravel()
accuracy_values = [None] * 20

for k in range(20):

    k_value = k+1

    neighbour = KNeighborsClassifier(n_neighbors = k_value, weights='uniform', algorithm='auto')

    neighbour.fit(x_train, y_train) 

    predict_y = neighbour.predict(x_test)

    accuracy_values[k] = accuracy_score(y_test,predict_y)*100

    print ("Accuracy = ", accuracy_values[k],"for K-Value = ",k_value)