import numpy as np

import pandas as pd

import os, sys

from sklearn.preprocessing import MinMaxScaler

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

df = pd.read_csv("/kaggle/input/parkinsons-data-set/parkinsons.data")

df.head()

features=df.loc[:,df.columns!='status'].values[:,1:]

labels=df.loc[:,'status'].values
print(labels[labels==1].shape[0], labels[labels==0].shape[0])

scaler=MinMaxScaler((-1,1))

x=scaler.fit_transform(features)

y=labels
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state=7)
def accuracy(prediction,actual):

    correct = 0

    not_correct = 0

    for i in range(len(prediction)):

        if prediction[i] == actual[i]:

            correct+=1

        else:

            not_correct+=1

    return (correct*100)/(correct+not_correct)





def metrics(prediction,actual):

    tp = 0

    tn = 0

    fp = 0

    fn = 0

    for i in range(len(prediction)):

        if prediction[i] == actual[i] and actual[i]==1:

            tp+=1

        if prediction[i] == actual[i] and actual[i]==0:

            tn+=1

        if prediction[i] != actual[i] and actual[i]==0:

            fp+=1

        if prediction[i] != actual[i] and actual[i]==1:

            fn+=1

    metrics = {'Precision':(tp/(tp+fp+tn+fn)),'Recall':(tp/(tp+fn)),'F1':(2*(tp/(tp+fp+tn+fn))*(tp/(tp+fn)))/((tp/(tp+fp+tn+fn))+(tp/(tp+fn)))}

    return (metrics)



from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier
#Logistic Regression

clf=LogisticRegression()

clf.fit(x_train, y_train)

preds=clf.predict(x_test)

print('accuracy:',accuracy(y_test.tolist(), preds.tolist()), '%')

print(metrics(y_test.tolist(), preds.tolist()))
#Random Forest

clf=RandomForestClassifier()

clf.fit(x_train, y_train)

preds=clf.predict(x_test)

print('accuracy:',accuracy(y_test.tolist(), preds.tolist()), '%')

print(metrics(y_test.tolist(), preds.tolist()))
#Support Vector Machine

clf=SVC()

clf.fit(x_train, y_train)

preds=clf.predict(x_test)

print('accuracy:',accuracy(y_test.tolist(), preds.tolist()), '%')

print(metrics(y_test.tolist(), preds.tolist()))
#Decision Tree

clf=DecisionTreeClassifier()

clf.fit(x_train, y_train)

preds=clf.predict(x_test)

print('accuracy:',accuracy(y_test.tolist(), preds.tolist()), '%')

print(metrics(y_test.tolist(), preds.tolist()))
#K-Nearest Neighbors



clf=KNeighborsClassifier()

clf.fit(x_train, y_train)

preds=clf.predict(x_test)

print('accuracy:',accuracy(y_test.tolist(), preds.tolist()), '%')

print(metrics(y_test.tolist(), preds.tolist()))



import matplotlib.pyplot as plt



names = ['Logistic Regression', 'Random Forest', 'SVM', 'Decision Tree', 'KNN']

values = [87.17948, 92.3076, 87.17948, 87.1794, 97.4358]



plt.figure(figsize=(9, 3))





plt.plot(names, values)



plt.show()

model=XGBClassifier()

model.fit(x_train,y_train)
y_pred=model.predict(x_test)

print(accuracy_score(y_test, y_pred)*100)