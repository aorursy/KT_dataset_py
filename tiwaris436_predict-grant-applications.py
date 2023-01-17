# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/unimelb/unimelb_training.csv",low_memory=False)

data.head()
len(list(data))
data.shape
data.isna().sum()
data1 = data.iloc[:,1:40] 

print(type(data1))
data1.isna().sum()
data2= data1.drop(['Country.of.Birth.1','Sponsor.Code','Grant.Category.Code','Start.date','Contract.Value.Band...see.note.A','Home.Language.1','With.PHD.1','No..of.Years.in.Uni.at.Time.of.Grant.1'],axis =1)
data2.shape
data2.isna().sum()
data3 = data2.dropna()
data3.isna().sum()
data3.shape
data3.head()
data3.dtypes
data3['Role.1'].unique()
data3['Role.1'] = data3['Role.1'].astype('category').cat.codes
data3['Grant.Status'].value_counts()
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
model = LogisticRegression()

X = data3.drop(['Grant.Status'],axis =1)

Y = data3['Grant.Status']
data3.shape
len(Y)
X.shape
X.head()
X_train, X_test, Y_train , Y_test = train_test_split(X,Y,test_size =0.2, random_state=42)
model.fit(X_train,Y_train)
pred = model.predict(X_test)
from sklearn.metrics import classification_report 

print(classification_report(Y_test, pred))
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

classifier.fit(X_train, Y_train)
# Predicting the Test set results

y_pred = classifier.predict(X_test)
print(classification_report(Y_test, y_pred))
from sklearn.ensemble import RandomForestClassifier
random_forest1 = RandomForestClassifier(n_estimators=15, max_depth=15)
random_forest1.fit(X, Y)
pred2 = random_forest1.predict(X_test)
print(classification_report(Y_test, pred2))