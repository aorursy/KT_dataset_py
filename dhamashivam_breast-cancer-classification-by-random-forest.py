import pandas as pd
import numpy as np
dataset=pd.read_csv("../input/data.csv")

dataset.head()
dataset.shape
dataset.info()
dataset.describe()
dataset.isnull().values.any()
#the last column has null values. remove it.
X=dataset.iloc[:,2:-1].values
y=dataset.iloc[:,1].values
from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
y=lb.fit_transform(y)
y
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
p=pd.DataFrame(X_train)
p.iloc[:,[29]].isnull().values.any()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=28)
rf.fit(X_train,y_train)
y_pred=rf.predict(X_test)
print('Accuracy is ',rf.score(X_test,y_test))
y_pred
y_test
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm
from sklearn.metrics import precision_recall_fscore_support as score

precision, recall, fscore, support = score(y_test, y_pred)

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))
