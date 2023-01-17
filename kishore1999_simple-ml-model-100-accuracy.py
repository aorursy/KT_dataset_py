import numpy as np 

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score,classification_report
data = pd.read_csv("../input/iris/Iris.csv")

data.head()
data=data.drop('Id',axis=1)

X=data.drop('Species',axis=1)

target=data['Species']
X_train,X_test,y_train,y_test=train_test_split(X,target,test_size=0.2)

X_train.head()
rf = RandomForestClassifier()

rf.fit(X_train,y_train)

predicted = rf.predict(X_test)

accuracy =  accuracy_score(y_test,predicted)

print('Accuracy:',accuracy)

print(classification_report(y_test,predicted))