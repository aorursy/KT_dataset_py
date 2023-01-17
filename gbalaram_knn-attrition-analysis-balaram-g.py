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

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix

from sklearn.metrics import f1_score

from sklearn.metrics import accuracy_score

df = pd.read_csv("../input/HR-Employee-Attrition.csv",index_col ="Age")



df.head(5)
df.shape
df.drop(["DailyRate","HourlyRate", "MonthlyRate"],axis=1,inplace = True)
df.drop(["EmployeeNumber","Over18"],axis=1,inplace = True)


df['Age']=pd.to_numeric(df.Age)

df.dtypes
Attr_num = {"Attrition" :{"Yes": 1, "No": 0}}
df.replace(Attr_num, inplace=True)
df
df_col_numeric = df.select_dtypes(include=np.number).columns
df_col_category =  df.select_dtypes(exclude=np.number).columns
df[df_col_category]
df_category_onehot = pd.get_dummies(df[df_col_category])
df_category_onehot
df_ready_model = pd.concat([df[df_col_numeric],df_category_onehot], axis = 1)
df_ready_model
X = df_ready_model.loc[:, df_ready_model.columns != 'Attrition']
Y = df_ready_model.loc[:,["Attrition"]]
X
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=123)

print("Training feature set size X:",X_train.shape)

print("Test feature set size X:",X_test.shape)

print("Training variable set size Y:",y_train.shape)

print("Test variable set size Y:",y_test.shape)
sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)
len(Y)
import math

math.sqrt(len(y_test))
classifier = KNeighborsClassifier(n_neighbors=21, p=2,metric='euclidean')
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)

y_pred
cm = confusion_matrix(y_test,y_pred)

print(cm)
print(f1_score(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
from sklearn import metrics

print("Train set Accuracy: ", metrics.accuracy_score(y_train, classifier.predict(X_train)))

print("Test set Accuracy: ", metrics.accuracy_score(y_test, y_test))