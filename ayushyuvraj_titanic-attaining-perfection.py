# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



# MACHINE LEARNING



###############################################################################



# PHASE 1 - PREPROCESSIG THE GIVEN DATASET



###############################################################################



# Importing the libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



#Importing the Dataset

dataset = pd.read_csv("../input/train.csv")

#dataset['Family'] = dataset['SibSp'] + dataset['Parch'] 

dataset.Embarked = dataset.Embarked.fillna("Q")

X = dataset.loc[:,('Pclass', 'Sex','Embarked', 'SibSp', 'Parch', 'Age', 'Fare') ].values

y = dataset.loc[:, ('Survived')].values
# Replacing numerical missing data with mean of the column

from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)

X[:, 3:6] = imputer.fit_transform(X[:,3:6])

X
#Encoding Categorical Data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X = LabelEncoder()

X[:,0] = labelencoder_X.fit_transform(X[:,0])

labelencoder_X = LabelEncoder()

X[:,1] = labelencoder_X.fit_transform(X[:,1])

labelencoder_X = LabelEncoder()

X[:,2] = labelencoder_X.fit_transform(X[:,2])

onehotencoder = OneHotEncoder(categorical_features=[0,1,2])

X = onehotencoder.fit_transform(X).toarray()
#Avoiding Dummy Variable Trap

X = X[:,1:]
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
#Feature Scaling

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)
# 1. Logistic Regression



# Fitting Logistic Regression to the Training set

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0)

classifier.fit(X_train, y_train)



# Predicting the test results

y_pred = classifier.predict(X_test)



# Making the confusion matrix

from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, y_pred)

print(classification_report(y_test, y_pred))

cm = (cm[0,0]+cm[1,1])/len(X_test)

cm
# Fitting kNN to the Training set

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier()

classifier.fit(X_train, y_train)



# Predicting the test results

y_pred = classifier.predict(X_test)



# Making the confusion matrix

from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, y_pred)

print(classification_report(y_test, y_pred))

cm = (cm[0,0]+cm[1,1])/len(X_test)

cm
# 3. SVM



# Fitting SVM to the Training set

from sklearn.svm import SVC

classifier = SVC(kernel = 'linear', random_state = 0)

classifier.fit(X_train, y_train)



# Predicting the Test set results

y_pred = classifier.predict(X_test)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

cm = (cm[0,0]+cm[1,1])/len(X_test)

cm
# 4. Kernel SVM



# Fitting Kernel SVM to the Training set

from sklearn.svm import SVC

classifier = SVC(kernel = 'rbf', random_state = 0)

classifier.fit(X_train, y_train)



# Predicting the Test set results

y_pred = classifier.predict(X_test)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

cm = (cm[0,0]+cm[1,1])/len(X_test)

cm
# 5. Naive Bayes



# Fitting Naive Bayes to the Training set

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(X_train, y_train)



# Predicting the Test set results

y_pred = classifier.predict(X_test)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

cm = (cm[0,0]+cm[1,1])/len(X_test)

cm
# 6. Decision Tree



# Fitting Decision Tree Classification to the Training set

from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

classifier.fit(X_train, y_train)



# Predicting the Test set results

y_pred = classifier.predict(X_test)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

cm = (cm[0,0]+cm[1,1])/len(X_test)

cm
# 7. Random Forest



# Fitting Random Forest Classification to the Training set

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 100, criterion = 'gini', random_state = 0)

classifier.fit(X_train, y_train)



# Predicting the Test set results

y_pred = classifier.predict(X_test)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

cm = (cm[0,0]+cm[1,1])/len(X_test)

cm
# 8. Artificial Neural Network



# Importing the Keras libraries and packages

import keras

from keras.models import Sequential

from keras.layers import Dense



# Initialising the ANN

classifier = Sequential()



# Adding the input layer and the first hidden layer

classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))



# Adding the second hidden layer

classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))



# Adding the output layer

classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))



# Compiling the ANN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



# Fitting the ANN to the Training set

classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 500)



# Predicting the Test set results

y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
# 9. XG Boost



# Fitting XGBoost to the Training set

from xgboost import XGBClassifier

classifier = XGBClassifier()

classifier.fit(X_train, y_train)



# Predicting the Test set results

y_pred = classifier.predict(X_test)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

cm = (cm[0,0]+cm[1,1])/len(X_test)

cm