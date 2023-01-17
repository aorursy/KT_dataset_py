#reference:

#https://www.kaggle.com/nirajvermafcb/support-vector-machine-detail-analysis/notebook



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/voice.csv')

df.head()
#Checking the correlation between each feature

df.corr()
#Checking whether there is any null values

df.isnull().sum()
#Checking shape of the dataset

df.shape
#print the target variable

print("Total number of labels: {}".format(df.shape[0]))

print("Number of male: {}".format(df[df.label == 'male'].shape[0]))

print("Number of female: {}".format(df[df.label == 'female'].shape[0]))
#Separating features and labels

X=df.iloc[:, :-1]

y=df.iloc[:,-1]
#Converting string value to int type for labels

from sklearn.preprocessing import LabelEncoder

gender_encoder = LabelEncoder()

y = gender_encoder.fit_transform(y)

y
#Data Standardisation

# Scale the data to be between -1 and 1

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X)

X = scaler.transform(X)
#Splitting dataset into training set and testing set for better generalisation

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
#Running SVM with default hyperparameter

from sklearn.svm import SVC

from sklearn import metrics

svc=SVC()

svc.fit(X_train,y_train)

y_pred=svc.predict(X_test)

print('Accuracy Score with default SVM:')

print(metrics.accuracy_score(y_test,y_pred))
#Running SVM with default Linear kernel

svc=SVC(kernel='linear')

svc.fit(X_train,y_train)

y_pred=svc.predict(X_test)

print('Accuracy Score with default Linear kerel SVM:')

print(metrics.accuracy_score(y_test,y_pred))
#Running SVM with RBF kernel

svc=SVC(kernel='rbf')

svc.fit(X_train,y_train)

y_pred=svc.predict(X_test)

print('Accuracy Score with RBF kernel SVM:')

print(metrics.accuracy_score(y_test,y_pred))

#Running SVM with Polynomial kernel

svc=SVC(kernel='poly')

svc.fit(X_train,y_train)

y_pred=svc.predict(X_test)

print('Accuracy Score with Polynomial kernel SVM:')

print(metrics.accuracy_score(y_test,y_pred))
#Performing K-fold cross validation with different kernels

from sklearn.model_selection import cross_val_score

svc=SVC(kernel='linear')

scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy') #cv is cross validation

print('CV on Linear kernel::',scores)
print(scores.mean())
from sklearn.cross_validation import cross_val_score

svc=SVC(kernel='rbf')

scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy') #cv is cross validation

print('CV on rbf kernel::',scores)
print(scores.mean())
from sklearn.cross_validation import cross_val_score

svc=SVC(kernel='poly')

scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy') #cv is cross validation

print('CV on Polynomial kernel::',scores)
print(scores.mean())
C_range=list(range(1,26))

acc_score=[]

for c in C_range:

    svc = SVC(kernel='linear', C=c)

    scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')

    acc_score.append(scores.mean())

print('accuracy score::',acc_score)

# plot the value of C for SVM (x-axis) versus the cross-validated accuracy (y-axis)

import matplotlib.pyplot as plt

%matplotlib inline

C_values=list(range(1,26))

plt.plot(C_values,acc_score)

plt.xticks(np.arange(0,27,2))

plt.xlabel('Value of C for SVC')

plt.ylabel('Cross-Validated Accuracy')
C_range=list(np.arange(0.1,6,0.1))

acc_score=[]

for c in C_range:

    svc = SVC(kernel='linear', C=c)

    scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')

    acc_score.append(scores.mean())

print('accuracy score::',acc_score)
# plot the value of C for SVM (x-axis) versus the cross-validated accuracy (y-axis)

import matplotlib.pyplot as plt

%matplotlib inline

C_values=list(np.arange(0.1,6,0.1))

plt.plot(C_values,acc_score)

plt.xticks(np.arange(0.0,6,0.3))

plt.xlabel('Value of C for SVC ')

plt.ylabel('Cross-Validated Accuracy')
gamma_range=[0.0001,0.001,0.01,0.1,1,10,100]

acc_score=[]

for g in gamma_range:

    svc = SVC(kernel='rbf', gamma=g)

    scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')

    acc_score.append(scores.mean())

print('accuracy score using rbf kernel and gamma::',acc_score)
# plot the value of C for SVM (x-axis) versus the cross-validated accuracy (y-axis)

import matplotlib.pyplot as plt

%matplotlib inline

gamma_range=[0.0001,0.001,0.01,0.1,1,10,100]

plt.plot(gamma_range,acc_score)

plt.xlabel('Value of gamma for SVC ')

plt.xticks(np.arange(0.0001,100,5))

plt.ylabel('Cross-Validated Accuracy')
gamma_range=[0.0001,0.001,0.01,0.1]

acc_score=[]

for g in gamma_range:

    svc = SVC(kernel='rbf', gamma=g)

    scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')

    acc_score.append(scores.mean())

print(acc_score)
# plot the value of C for SVM (x-axis) versus the cross-validated accuracy (y-axis)

import matplotlib.pyplot as plt

%matplotlib inline

gamma_range=[0.0001,0.001,0.01,0.1]

plt.plot(gamma_range,acc_score)

plt.xlabel('Value of gamma for SVC ')

plt.ylabel('Cross-Validated Accuracy')
gamma_range=[0.01,0.02,0.03,0.04,0.05]

acc_score=[]

for g in gamma_range:

    svc = SVC(kernel='rbf', gamma=g)

    scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')

    acc_score.append(scores.mean())

print(acc_score)
import matplotlib.pyplot as plt

%matplotlib inline

gamma_range=[0.01,0.02,0.03,0.04,0.05]

# plot the value of C for SVM (x-axis) versus the cross-validated accuracy (y-axis)

plt.plot(gamma_range,acc_score)

plt.xlabel('Value of gamma for SVC ')

plt.ylabel('Cross-Validated Accuracy')
#Taking polynomial kernel with different degree

degree=[2,3,4,5,6]

acc_score=[]

for d in degree:

    svc = SVC(kernel='poly', degree=d)

    scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')

    acc_score.append(scores.mean())

print('accuracy score using poly kernel and degree::',acc_score)
import matplotlib.pyplot as plt

%matplotlib inline

degree=[2,3,4,5,6]

# plot the value of C for SVM (x-axis) versus the cross-validated accuracy (y-axis)

plt.plot(degree,acc_score,color='r')

plt.xlabel('degrees for SVC ')

plt.ylabel('Cross-Validated Accuracy')
from sklearn.svm import SVC

svc= SVC(kernel='linear',C=0.1)

svc.fit(X_train,y_train)

y_predict=svc.predict(X_test)

accuracy_score= metrics.accuracy_score(y_test,y_predict)

print('accuracy_score using linear kernel and 0.1 C::',accuracy_score)
#With K-fold cross validation(where K=10)

from sklearn.cross_validation import cross_val_score

svc=SVC(kernel='linear',C=0.1)

scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')

print('cross validation score using linear kernel,0.1 C and k fold::',scores)
print(scores.mean())
#Now performing SVM by taking hyperparameter gamma=0.01 and kernel as rbf

from sklearn.svm import SVC

svc= SVC(kernel='rbf',gamma=0.01)

svc.fit(X_train,y_train)

y_predict=svc.predict(X_test)

metrics.accuracy_score(y_test,y_predict)
#With K-fold cross validation(where K=10)

svc=SVC(kernel='linear',gamma=0.01)

scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')

print(scores)

print(scores.mean())
#Now performing SVM by taking hyperparameter degree=3 and kernel as poly

from sklearn.svm import SVC

svc= SVC(kernel='poly',degree=3)

svc.fit(X_train,y_train)

y_predict=svc.predict(X_test)

accuracy_score= metrics.accuracy_score(y_test,y_predict)

print(accuracy_score)
#With K-fold cross validation(where K=10)

svc=SVC(kernel='poly',degree=3)

scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')

print(scores)

print(scores.mean())
#Let us perform Grid search technique to find the best parameter

from sklearn.svm import SVC

svm_model= SVC()



tuned_parameters = {

 'C': (np.arange(0.1,1,0.1)) , 'kernel': ['linear'],

 'C': (np.arange(0.1,1,0.1)) , 'gamma': [0.01,0.02,0.03,0.04,0.05], 'kernel': ['rbf'],

 'degree': [2,3,4] ,'gamma':[0.01,0.02,0.03,0.04,0.05], 'C':(np.arange(0.1,1,0.1)) , 'kernel':['poly']

                   }



from sklearn.model_selection import GridSearchCV

model_svm = GridSearchCV(svm_model, tuned_parameters,cv=10,scoring='accuracy')
model_svm.fit(X_train, y_train)

print(model_svm.best_score_)
print(model_svm.best_params_)
#predicting the model

y_pred= model_svm.predict(X_test)

print(metrics.accuracy_score(y_pred,y_test))