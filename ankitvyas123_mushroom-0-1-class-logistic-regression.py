import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

%matplotlib inline



import seaborn as sns
data = pd.read_csv("../input/mushrooms.csv")
data.head()
data.shape
data['class'].value_counts()
for i in range(len(data.columns)):

    print(data.iloc[:,i].value_counts())
class_mapping = {'e':0,'p':1}
data['class']=data['class'].map(class_mapping)
data2 = pd.get_dummies(data)
data2.head()
# use logistic regression
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(data2.iloc[:,1:],data2.iloc[:,0],test_size=0.2)
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=0.2)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train,y_train)
y_pred_train = lr.predict(x_train)
y_pred_val = lr.predict(x_val)
y_pred_test = lr.predict(x_test)
lr.score(x_train,y_train)
lr.score(x_val,y_val)
lr.score(x_test,y_test)
from sklearn.metrics import classification_report
print(classification_report(y_train,y_pred_train))
print(classification_report(y_test,y_pred_test))
print(classification_report(y_val,y_pred_val))
from sklearn.metrics import confusion_matrix
confusion_matrix(y_train,y_pred_train)
confusion_matrix(y_test,y_pred_test)
confusion_matrix(y_val,y_pred_val)
# Everything Perfect!