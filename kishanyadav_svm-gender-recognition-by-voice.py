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
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
df = pd.read_csv("../input/voice.csv")
df.shape
df.head()
df.corr()
df.isnull().sum()
print("Number of male: {}".format(df[df.label == 'male'].shape[0]))
print("Number of female: {}".format(df[df.label == 'female'].shape[0]))
# No. of male and female are same

df.shape
# Sperating data

X = df.iloc[:,:-1]

X.head()

y = df.iloc[:,20]

y.head()
from sklearn.preprocessing import StandardScaler,LabelEncoder

gender_encoder = LabelEncoder()

y_label = gender_encoder.fit_transform(y)
# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X = sc_X.fit_transform(X)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y_label,test_size=0.25,random_state=1)
# Default HyperParameter :-

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score

classification = SVC(gamma="auto")

classification.fit(X_train,y_train)



y_pred = classification.predict(X_test)

cnf_matrix = confusion_matrix(y_test,y_pred)

print("Confusion matrices: ")

print(cnf_matrix)

print("Precision:",precision_score(y_test, y_pred))

print("Recall:",recall_score(y_test, y_pred))

print("Accuracy:",accuracy_score(y_test, y_pred))
# Linear Kernel :-

# Equation of Polynomial Kernel : K(x, xi) = sum(x * xi)

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

classification = SVC(kernel = "linear")

classification.fit(X_train,y_train)



y_pred = classification.predict(X_test)

print("Confusion matrices: ")

print(cnf_matrix)

print("Precision:",precision_score(y_test, y_pred))

print("Recall:",recall_score(y_test, y_pred))

print("Accuracy:",accuracy_score(y_test, y_pred))
# Polynomial  Kernel :-

# Equation of Polynomial Kernel : K(x,xi) = 1 + sum(x * xi)^d

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

classification = SVC(kernel = "poly",gamma='auto')

classification.fit(X_train,y_train)



y_pred = classification.predict(X_test)

print("Confusion matrices: ")

print(cnf_matrix)

print("Precision:",precision_score(y_test, y_pred))

print("Recall:",recall_score(y_test, y_pred))

print("Accuracy:",accuracy_score(y_test, y_pred))
# rbf Kernel

# Equation of radial basis function ==== K(x,xi) = exp(-gamma * sum((x – xi^2))

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

classification = SVC(kernel = "rbf",gamma = 'auto')

classification.fit(X_train,y_train)



y_pred = classification.predict(X_test)

print("Confusion matrices: ")

print(cnf_matrix)

print("Precision:",precision_score(y_test, y_pred))

print("Recall:",recall_score(y_test, y_pred))

print("Accuracy:",accuracy_score(y_test, y_pred))
# cv on linear

from sklearn.svm import SVC

from sklearn.model_selection  import cross_val_score

classification_svc = SVC(kernel = "linear")

score = cross_val_score(classification_svc,X,y_label,cv=10, scoring="accuracy")

print(score)



print()

print("Mean score of linear kernel : ")

print(score.mean())
# cv on poly

classification_svc = SVC(kernel = "poly",gamma="auto")

score = cross_val_score(classification_svc,X,y_label,cv=10, scoring="accuracy")

print(score)

print()

print("Mean score of poly kernel : ")

print(score.mean())
# cv on rbf



classification_svc = SVC(kernel = "rbf",gamma="auto")

score = cross_val_score(classification_svc,X,y_label,cv=10, scoring="accuracy")

print(score)

print()

print("Mean score of rbf kernel : ")

print(score.mean())
# Checking accuracy on C value 



C_range=list(range(1,26))

acc_score=[]

for c in C_range:

    svc = SVC(kernel='linear', C=c)

    scores = cross_val_score(svc, X, y_label, cv=10, scoring='accuracy')

    acc_score.append(scores.mean())

print(acc_score)  
C_values=list(range(1,26))

plt.plot(C_values,acc_score)

plt.xticks(np.arange(0,27,2))

plt.xlabel('Value of C for SVC')

plt.ylabel('Cross-Validated Accuracy')
from sklearn.model_selection import validation_curve

param_range = [ 0.01, 0.1, 1.0, 10]

train_scores, valid_scores = validation_curve(SVC(kernel='linear'), X, y_label,param_name="gamma", param_range=param_range,cv=5)
print(train_scores)
print(valid_scores)
# Time for GridSearch

from sklearn.model_selection import GridSearchCV

svm_model = SVC()

param_range = [ 0.1, 1.0, 10]

tuned_parameters = {

 'C': (np.arange(0.1,1,0.1)) , 'kernel': ['linear'],

 'C': (np.arange(0.1,1,0.1)) , 'gamma': [0.01,0.02,0.03,0.04,0.05], 'kernel': ['rbf'],

 'degree': [2,3,4] ,'gamma':[0.01,0.02,0.03,0.04,0.05], 'C':(np.arange(0.1,1,0.1)) , 'kernel':['poly']}
model_svm = GridSearchCV(svm_model, tuned_parameters,cv=10,scoring='accuracy')

model_svm.fit(X_train,y_train)
print("Acurracy : ",model_svm.best_score_)
print('Best Parameters : ',(model_svm.best_params_))
y_pred = model_svm.predict(X_test)
print('Predicted value: ',  y_pred)
print('Test Accuracy: %.7f' % model_svm.score(X_test, y_test))