# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import seaborn as sns



import matplotlib.pyplot as plt



%matplotlib inline
df = pd.read_csv('../input/voice.csv')

df.head()
df.corr()
df.isnull().sum()
df.shape
print("Total number of labels: {}".format(df.shape[0]))

print("Number of male: {}".format(df[df.label == 'male'].shape[0]))

print("Number of female: {}".format(df[df.label == 'female'].shape[0]))
df.shape
X=df.iloc[:, :-1]

X.head()
from sklearn.preprocessing import LabelEncoder

y=df.iloc[:,-1]



# Encode label category

# male -> 1

# female -> 0



gender_encoder = LabelEncoder()

y = gender_encoder.fit_transform(y)

y
# Scale the data to be between -1 and 1

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X)

X = scaler.transform(X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
from sklearn.svm import SVC

from sklearn import metrics

svc=SVC() #Default hyperparameters

svc.fit(X_train,y_train)

y_pred=svc.predict(X_test)

print('Accuracy Score:')

print(metrics.accuracy_score(y_test,y_pred))
svc=SVC(kernel='linear')

svc.fit(X_train,y_train)

y_pred=svc.predict(X_test)

print('Accuracy Score:')

print(metrics.accuracy_score(y_test,y_pred))
svc=SVC(kernel='rbf')

svc.fit(X_train,y_train)

y_pred=svc.predict(X_test)

print('Accuracy Score:')

print(metrics.accuracy_score(y_test,y_pred))
svc=SVC(kernel='poly')

svc.fit(X_train,y_train)

y_pred=svc.predict(X_test)

print('Accuracy Score:')

print(metrics.accuracy_score(y_test,y_pred))
from sklearn.cross_validation import cross_val_score

svc=SVC(kernel='linear')

scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy') #cv is cross validation

print(scores)
print(scores.mean())
from sklearn.cross_validation import cross_val_score

svc=SVC(kernel='rbf')

scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy') #cv is cross validation

print(scores)
print(scores.mean())
from sklearn.cross_validation import cross_val_score

svc=SVC(kernel='poly')

scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy') #cv is cross validation

print(scores)
print(scores.mean())
C_range=list(range(1,26))

acc_score=[]

for c in C_range:

    svc = SVC(kernel='linear', C=c)

    scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')

    acc_score.append(scores.mean())

print(acc_score)    

    
import matplotlib.pyplot as plt

%matplotlib inline





C_values=list(range(1,26))

# plot the value of C for SVM (x-axis) versus the cross-validated accuracy (y-axis)

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

print(acc_score)    

    
import matplotlib.pyplot as plt

%matplotlib inline



C_values=list(np.arange(0.1,6,0.1))

# plot the value of C for SVM (x-axis) versus the cross-validated accuracy (y-axis)

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

print(acc_score)    

    
import matplotlib.pyplot as plt

%matplotlib inline



gamma_range=[0.0001,0.001,0.01,0.1,1,10,100]



# plot the value of C for SVM (x-axis) versus the cross-validated accuracy (y-axis)

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

    
import matplotlib.pyplot as plt

%matplotlib inline



gamma_range=[0.0001,0.001,0.01,0.1]



# plot the value of C for SVM (x-axis) versus the cross-validated accuracy (y-axis)

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
degree=[2,3,4,5,6]

acc_score=[]

for d in degree:

    svc = SVC(kernel='poly', degree=d)

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
from sklearn.svm import SVC

svc= SVC(kernel='linear',C=0.1)

svc.fit(X_train,y_train)

y_predict=svc.predict(X_test)

accuracy_score= metrics.accuracy_score(y_test,y_predict)

print(accuracy_score)
from sklearn.cross_validation import cross_val_score

svc=SVC(kernel='linear',C=0.1)

scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')

print(scores)
print(scores.mean())
from sklearn.svm import SVC

svc= SVC(kernel='rbf',gamma=0.01)

svc.fit(X_train,y_train)

y_predict=svc.predict(X_test)

metrics.accuracy_score(y_test,y_predict)
svc=SVC(kernel='linear',gamma=0.01)

scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')

print(scores)

print(scores.mean())
from sklearn.svm import SVC

svc= SVC(kernel='poly',degree=3)

svc.fit(X_train,y_train)

y_predict=svc.predict(X_test)

accuracy_score= metrics.accuracy_score(y_test,y_predict)

print(accuracy_score)
svc=SVC(kernel='poly',degree=3)

scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')

print(scores)

print(scores.mean())
from sklearn.svm import SVC

svm_model= SVC()
tuned_parameters = {

 'C': (np.arange(0.1,1,0.1)) , 'kernel': ['linear'],

 'C': (np.arange(0.1,1,0.1)) , 'gamma': [0.01,0.02,0.03,0.04,0.05], 'kernel': ['rbf'],

 'degree': [2,3,4] ,'gamma':[0.01,0.02,0.03,0.04,0.05], 'C':(np.arange(0.1,1,0.1)) , 'kernel':['poly']

                   }
from sklearn.grid_search import GridSearchCV



model_svm = GridSearchCV(svm_model, tuned_parameters,cv=10,scoring='accuracy')
model_svm.fit(X_train, y_train)

print(model_svm.best_score_)
print(model_svm.grid_scores_)
print(model_svm.best_params_)
y_pred= model_svm.predict(X_test)

print(metrics.accuracy_score(y_pred,y_test))