# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

df = pd.read_csv("../input/WA_Fn-UseC_-HR-Employee-Attrition.csv")

df.head()

df.pop('EmployeeNumber')

df.pop('Over18')

df.pop('StandardHours')

df.pop('EmployeeCount')



# Any results you write to the current directory are saved as output.



y =df['Attrition']

X = df

X.head()

X.pop('Attrition')

y.unique()



from sklearn import preprocessing

le = preprocessing.LabelBinarizer()



y = le.fit_transform(y)



y.shape



df.select_dtypes(['object'])



ind_BusinessTravel = pd.get_dummies(df['BusinessTravel'], prefix='BusinessTravel')

ind_Department = pd.get_dummies(df['Department'], prefix='Department')

ind_EducationField = pd.get_dummies(df['EducationField'], prefix='EducationField')

ind_Gender = pd.get_dummies(df['Gender'], prefix='Gender')

ind_JobRole = pd.get_dummies(df['JobRole'], prefix='JobRole')

ind_MaritalStatus = pd.get_dummies(df['MaritalStatus'], prefix='MaritalStatus')

ind_OverTime = pd.get_dummies(df['OverTime'], prefix='OverTime')





df1 = pd.concat([ind_BusinessTravel, ind_Department, ind_EducationField, ind_Gender, 

                 ind_JobRole, ind_MaritalStatus, ind_OverTime])



df1 = pd.concat([ind_BusinessTravel, ind_Department, ind_EducationField, ind_Gender, 

                 ind_JobRole, ind_MaritalStatus, ind_OverTime, df.select_dtypes(['int64'])], axis=1)

                

from sklearn.model_selection import train_test_split

                

X_train, X_test, y_train, y_test = train_test_split(df1, y)   

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state=42)

clf.fit(X_train, y_train)

                

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

accuracy_score(y_train, clf.predict(X_train))

print(classification_report(y_train, clf.predict(X_train)))

confusion_matrix(y_train, clf.predict(X_train))

accuracy_score(y_test, clf.predict(X_test))

print(classification_report(y_test, clf.predict(X_test)))

                

confusion_matrix(y_test, clf.predict(X_test))










