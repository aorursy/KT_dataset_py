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
data = pd.read_csv("../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")

data.head()
data.info()
import seaborn as sns

import matplotlib.pyplot as plt



#Using Pearson Correlation

plt.figure(figsize=(12,10))

cor = data.corr()

sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)

plt.show()
from sklearn.preprocessing import LabelEncoder



for i in range(len(data['quality'])):

    if(data['quality'].iloc[i] <= 6.5):

        data['quality'].iloc[i] = 0

    else:

        data['quality'].iloc[i] = 1

#Bad becomes 0 and good becomes 1 

data.head()

#print(data['quality'].values.tolist())
print("Value distribution of bad(0)/good(1) quality: \n{}".format(data['quality'].value_counts()))

# Creating a pairplot to visualize the similarities and especially difference between the quality

# sns.pairplot(data=data, hue='quality', palette='Set2')
from sklearn.model_selection import train_test_split



#Split dataset into training set and test set

X = data.drop('quality', axis = 1)

y = data['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)



print("Total number of examples " + str(len(data.index)))

print("Number of training set examples "+ str(len(X_train)))

print("Number of test set examples "+ str(len(X_test)))
from sklearn.preprocessing import StandardScaler

#Applying Standard scaling to get optimized result

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.fit_transform(X_test)
from sklearn.linear_model import SGDClassifier

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.metrics import accuracy_score



sgd = SGDClassifier(penalty=None)

sgd.fit(X_train, y_train)

pred_sgd = sgd.predict(X_test)



print('Confusion matrix: ')

print(confusion_matrix(y_test,pred_sgd))



print(classification_report(y_test, pred_sgd))



#print('Accuracy Score: {}'.format(accuracy_score(y_test,pred_sgd)))
from sklearn.svm import SVC



svc = SVC() #Default hyperparameters

svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)



print('Confusion matrix: ')

print(confusion_matrix(y_test,y_pred))

#sns.heatmap(confusion_matrix(y_test,y_pred), annot=True, cmap=plt.cm.Blues)

#plt.show()



print(classification_report(y_test, y_pred))



#print('Accuracy Score: {}'.format(accuracy_score(y_test,y_pred)))
svc=SVC(kernel='linear')

svc.fit(X_train,y_train)

y_pred=svc.predict(X_test)



print('Confusion matrix: ')

print(confusion_matrix(y_test,y_pred))

#sns.heatmap(confusion_matrix(y_test,y_pred), annot=True, cmap=plt.cm.Blues)

#plt.show()



print(classification_report(y_test, y_pred))



#print('Accuracy Score: {}'.format(accuracy_score(y_test,y_pred)))
svc=SVC(kernel='rbf')

svc.fit(X_train,y_train)

y_pred=svc.predict(X_test)



print('Confusion matrix: ')

print(confusion_matrix(y_test,y_pred))

#sns.heatmap(confusion_matrix(y_test,y_pred), annot=True, cmap=plt.cm.Blues)

#plt.show()



print(classification_report(y_test, y_pred))



#print('Accuracy Score: {}'.format(accuracy_score(y_test,y_pred)))
svc=SVC(kernel='poly')

svc.fit(X_train,y_train)

y_pred=svc.predict(X_test)



print('Confusion matrix: ')

print(confusion_matrix(y_test,y_pred))

#sns.heatmap(confusion_matrix(y_test,y_pred), annot=True, cmap=plt.cm.Blues)

#plt.show()



print(classification_report(y_test, y_pred))



#print('Accuracy Score: {}'.format(accuracy_score(y_test,y_pred)))
from sklearn.model_selection import GridSearchCV



svm_model= SVC()



tuned_parameters = {

    'C': [0.1,0.8,0.9,1,1.1,1.2,1.3,1.4],

    'kernel':['linear', 'rbf'],

    'gamma' :[0.1,0.8,0.9,1,1.1,1.2,1.3,1.4]

                   }



grid_svc = GridSearchCV(svm_model, tuned_parameters, scoring='accuracy', cv=10) # K = 10

grid_svc.fit(X_train, y_train)

#Best parameters for our svc model

grid_svc.best_params_
print(grid_svc.best_score_)
svc2 = SVC(C = 1.2, gamma =  0.9, kernel= 'rbf')

svc2.fit(X_train, y_train)

pred_svc2 = svc2.predict(X_test)

print(classification_report(y_test, pred_svc2))
from sklearn.model_selection import cross_val_score



degree=[2,3,4,5,6]

acc_score=[]

for d in degree:

    svc = SVC(kernel='poly', degree=d, gamma='scale')

    scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')

    acc_score.append(scores.mean())

print(acc_score)
import matplotlib.pyplot as plt

%matplotlib inline



degree=[2,3,4,5,6]



# plot the value of C for SVM (x-axis) versus the cross-validated accuracy (y-axis)

plt.plot(degree,acc_score,color='r')

plt.xlabel('degrees for SVC ')

plt.ylabel('Cross-Validated Accuracy')
'''import timeit



svc = SVC()



tuned_parameters = {

    'C':[0.1,0.8,0.9,1,1.1,1.2,1.3,1.4], 'kernel':['linear'],

    'gamma':[0.1,0.8,0.9,1,1.1,1.2,1.3,1.4], 'C':[0.1,0.8,0.9,1,1.1,1.2,1.3,1.4] , 'kernel':['rbf'],

    'degree': [2,3,4] ,'gamma':[0.1,0.8,0.9,1,1.1,1.2,1.3,1.4], 'C':[0.1,0.8,0.9,1,1.1,1.2,1.3,1.4] , 'kernel':['poly']

}



start = timeit.default_timer()

grid_svc = GridSearchCV(svc, tuned_parameters, scoring='accuracy', cv=10) # K = 10

grid_svc.fit(X_train, y_train)

#Best parameters for our svc model

print(grid_svc.best_params_)

stop = timeit.default_timer()

print('Time: ', stop - start)'''



# If we try to find best parameters in this way, GridSearchCV returns polynomial kernel(degree = 3) and different value of C and gamma

# instead of gaussian kernel(rbf) with C=1.2 and gamma=0.9. The problem is with the accuracy: 

# with polynomial kernel (and its parameters) the accuracy is around to 86% but with gaussian kernel is 90%.

# I prefered to keep separate cases to show how the accuracy varying with different degrees on polynomial kernel.

# A timer is been added to show the time of computation if someone wanted to try.