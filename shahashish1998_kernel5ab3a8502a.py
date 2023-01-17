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
#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing our cancer dataset
dataset = pd.read_excel('../input/BreastCancer_Prognostic_v1.xlsx',na_values=["?"])
X = dataset.iloc[:, 3:35].values
Y = dataset.iloc[:, 1:2].values


#taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer()
imputer = imputer.fit(X[:,31:32])
X[:,31:32] = imputer.transform(X[:,31:32])

'''X.plot(kind='density', subplots=True, layout=(5,7), sharex=False, legend=False, fontsize=1)
plt.show()'''

#Encoding categorical data values
from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
labelencoder_Y.fit(Y)
Y = labelencoder_Y.transform(Y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

'''Using Logistic Regression Algorithm to the Training Set
from sklearn.linear_model import LogisticRegression
logistic_classifier = LogisticRegression(random_state = 0)
logistic_classifier.fit(X_train, Y_train)

#Predicting the values of Y_pred
logistic_Y_pred = logistic_classifier.predict(X_test)

#Accuracy checking using Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
logistic_cm = confusion_matrix(Y_test, logistic_Y_pred)
print ('Logistic Regression Algorithm')
print ('Accuracy Score :',accuracy_score(Y_test, logistic_Y_pred))
print('Classification Report : ')
print (classification_report(Y_test, logistic_Y_pred))
'''

'''Using KNeighborsClassifier Method of neighbors class to use Nearest Neighbor algorithm
from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn_classifier.fit(X_train, Y_train)

#Predicting the values of Y_pred
KNN_Y_pred = knn_classifier.predict(X_test)

#Accuracy checking using Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
KNN_cm = confusion_matrix(Y_test, KNN_Y_pred)
print('Nearest Neighbor algorithm')
print ('Accuracy Score :',accuracy_score(Y_test, KNN_Y_pred))
print('Classification Report : ')
print (classification_report(Y_test, KNN_Y_pred))
'''

'''Using SVC method of svm class to use Support Vector Machine Algorithm
from sklearn.svm import SVC
svm_classifier = SVC(kernel = 'linear', random_state = 0)
svm_classifier.fit(X_train, Y_train)

#Predicting the values of Y_pred
SVM_Y_pred = svm_classifier.predict(X_test)

#Accuracy checking using Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
SVM_cm = confusion_matrix(Y_test, SVM_Y_pred)
print('Support Vector Machine Algorithm')
print ('Accuracy Score :',accuracy_score(Y_test, SVM_Y_pred))
print('Classification Report : ')
print (classification_report(Y_test, SVM_Y_pred))
'''

'''Using SVC method of svm class to use Kernel SVM Algorithm
from sklearn.svm import SVC
ksvm_classifier = SVC(kernel = 'rbf', random_state = 0)
ksvm_classifier.fit(X_train, Y_train)

#Predicting the values of Y_pred
Kernel_SVM_Y_pred = ksvm_classifier.predict(X_test)

#Accuracy checking using Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
Kernel_SVM_cm = confusion_matrix(Y_test, Kernel_SVM_Y_pred)
print('Kernel SVM Algorithm')
print ('Accuracy Score :',accuracy_score(Y_test, Kernel_SVM_Y_pred))
print('Classification Report : ')
print (classification_report(Y_test, Kernel_SVM_Y_pred))
'''

'''Using GaussianNB method of naive_bayes class to use Naïve Bayes Algorithm
from sklearn.naive_bayes import GaussianNB
gausian_classifier = GaussianNB()
gausian_classifier.fit(X_train, Y_train)

#Predicting the values of Y_pred
Gausian_Y_pred = gausian_classifier.predict(X_test)

#Accuracy checking using Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
Gausian_cm = confusion_matrix(Y_test, Gausian_Y_pred)
print('Naïve Bayes Algorithm')
print ('Accuracy Score :',accuracy_score(Y_test, Gausian_Y_pred))
print('Classification Report : ')
print (classification_report(Y_test, Gausian_Y_pred))
'''

#Using DecisionTreeClassifier of tree class to use Decision Tree Algorithm

from sklearn.tree import DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
dt_classifier.fit(X_train, Y_train)

#Predicting the values of Y_pred
DecisionTree_Y_pred = dt_classifier.predict(X_test)

#Accuracy checking using Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
DecisionTree_cm = confusion_matrix(Y_test, DecisionTree_Y_pred)
print('Decision Tree Algorithm')
print ('Accuracy Score :',accuracy_score(Y_test, DecisionTree_Y_pred))
print('Classification Report : ')
print (classification_report(Y_test, DecisionTree_Y_pred))
Y_pred = labelencoder_Y.inverse_transform(DecisionTree_Y_pred)
'''
#Using RandomForestClassifier method of ensemble class to use Random Forest Classification algorithm

from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
rf_classifier.fit(X_train, Y_train)

#Predicting the values of Y_pred
RandomForest_Y_pred = rf_classifier.predict(X_test)

#Accuracy checking using Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
RandomForest_cm = confusion_matrix(Y_test, RandomForest_Y_pred)
print('Random Forest Classification algorithm')
print ('Accuracy Score :',accuracy_score(Y_test, RandomForest_Y_pred))
print('Classification Report : ')
print (classification_report(Y_test, RandomForest_Y_pred))
'''
