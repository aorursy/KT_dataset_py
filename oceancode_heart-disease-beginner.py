import numpy as np

import pandas as pd

data= pd.read_csv("../input/heart.csv") 
data.head()
data.info()
data.isnull().sum()
data.dtypes
cp=pd.get_dummies(data['cp'],prefix='cp', drop_first= True)

exang=pd.get_dummies(data['exang'],prefix='exang', drop_first= True)

slope=pd.get_dummies(data['slope'],prefix='slope', drop_first=True)
new_data= pd.concat([data,cp,exang,slope], axis=1)

new_data.head()
new_data.drop(['cp','exang','slope'], axis= 1, inplace= True)

new_data.head()
y=new_data['target']

X=new_data.drop(['target'], axis= 1)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train, y_test= train_test_split(X,y, test_size=0.2, random_state= 2)
from sklearn.linear_model import LogisticRegression

lr= LogisticRegression()

lr.fit(X_train,y_train)

lr.score(X_test, y_test)
from sklearn.neighbors import KNeighborsClassifier

knn= KNeighborsClassifier()

knn.fit(X_train,y_train)

knn.score(X_test, y_test)

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=1)

dt.fit(X_train, y_train)

dt.score(X_test, y_test)
from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier()

gbc.fit(X_train, y_train)

gbc.score(X_test, y_test)
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(X_train, y_train)

nb.score(X_test, y_test)
from sklearn.ensemble import RandomForestClassifier

for i in range(1, 10):

    rfc = RandomForestClassifier(n_estimators=i)

    rfc.fit(X_train, y_train)

    print('n_estimators : ', i, "score : ", rfc.score(X_test, y_test), end="\n")
from sklearn.svm import SVC

svc = SVC(kernel='linear')

svc.fit(X_train, y_train)

svc.score(X_test, y_test)