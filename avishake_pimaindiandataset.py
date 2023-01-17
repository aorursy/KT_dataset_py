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
df = pd.read_csv("/kaggle/input/pima-indians-diabetes-database/diabetes.csv")
df.head()
df.isnull().sum()
X = df
y = df.Outcome
X.drop("Outcome",axis = 1,inplace = True)
X
y
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
from sklearn.model_selection import train_test_split
X,X_test,y,y_test = train_test_split(X,y,test_size = 0.2,random_state = 1)
lr.fit(X,y)
pred = lr.predict(X_test)
from sklearn.metrics import r2_score

r2_score(pred,y_test)
from sklearn.metrics import accuracy_score
accuracy_score(pred,y_test)*100
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 500, random_state = 1)

rf.fit(X,y)
pred = rf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,pred)*100
from xgboost import XGBClassifier
xgb = XGBClassifier(random_state = 1,n_estimators = 100)
xgb.fit(X,y)
pred = xgb.predict(X_test)
accuracy_score(y_test,pred)*100
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors = 50)
model.fit(X,y)
pred = model.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(pred,y_test)*100
from sklearn.svm import SVC
model2 = SVC()
model2.fit(X,y)
pred3 = model2.predict(X_test)
from sklearn.metrics import accuracy_score

accuracy_score(y_test,pred3)
X.columns
X.corr()
df.columns
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit_transform(X)
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(random_state = 1, n_estimators = 1000)

rfc.fit(X,y)
pred5 = rfc.predict(X_test)
accuracy_score(y_test,pred5)*100