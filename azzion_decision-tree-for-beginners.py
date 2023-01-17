import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

df = pd.read_csv('../input/train.csv')
df.head()
df.isnull().sum()
df = df.drop(['Cabin','Age'],axis=1)
df = df.drop(df.loc[df['Embarked'].isnull()].index,axis=0)
df.isnull().sum().max()

y = df['Survived']
X = df
X = X.drop(['Survived','PassengerId','Name','Ticket'],axis=1)
X = pd.get_dummies(X)
X.head(2)
feat_names = X.columns
targ_names = ['Yes','No']
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,y,random_state=42,test_size=.1)
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.metrics import accuracy_score,recall_score,precision_score,confusion_matrix,f1_score
import matplotlib.pyplot as plt
import seaborn as sns

clf = DecisionTreeClassifier(max_depth=3).fit(X_train,Y_train)
print("Training:"+str(clf.score(X_train,Y_train)))
print("Test:"+str(clf.score(X_test,Y_test)))
pred = clf.predict(X_train)
confusion_matrix = confusion_matrix(y_true=Y_train,y_pred=pred)

sns.heatmap(confusion_matrix,annot=True,annot_kws={"size":16})
plt.show()

confusion_matrix


print("precision score : "+str(precision_score(Y_train,pred))) # tp/tp+fp
print("accuracy score : "+str(accuracy_score(Y_train,pred))) # total correct 
print("recall score : "+str(recall_score(Y_train,pred)))   # tp/tp+fn
print("f1 score : "+str(f1_score(Y_train,pred))) 

import graphviz

data = export_graphviz(clf,out_file=None,feature_names=feat_names,class_names=targ_names,   
                         filled=True, rounded=True,  
                         special_characters=True)
graph = graphviz.Source(data)
graph
# predicting and checking accuracy 
predict = pd.read_csv("../input/test.csv")
predict.isnull().sum()
predict = predict.drop(['Cabin','Age'],axis=1)
predict = predict.drop(predict.loc[predict['Fare'].isnull()].index,axis=0)
predict = predict.drop(['Name','Ticket','PassengerId'],axis=1)
predict = predict.drop(['PassengerId'],axis=1)
predict = pd.get_dummies(predict)
predicted_values = clf.predict(predict)
predicted_table = pd.DataFrame(predicted_values)
predicted_table.head()