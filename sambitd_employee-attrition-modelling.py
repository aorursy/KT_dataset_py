# Importing the libraries and the dataset 

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset and pre-processing
dataset = pd.read_csv('../input/MFG10YearTerminationData.csv')
mapping = {'ACTIVE': 1, 'TERMINATED': 0}   # Convert the categorical object into numerical
dataset = dataset.replace({'STATUS': mapping})
y = dataset.STATUS
# Create the predictors dataset

features = ['age', 'length_of_service', 'gender_full', 'STATUS_YEAR', 'BUSINESS_UNIT']
X = dataset[features]
# The gender, business unit are nominal, so they will
# be exploded instead of being converted to ordinal values

dummy_cols = ['gender_full', 'BUSINESS_UNIT']
X = pd.get_dummies(X, columns=dummy_cols)
#Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_predlc = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cmlg = confusion_matrix(y_test, y_predlc)

score = classifier.score(X_test, y_test)
print('Logistic model score is %0.4f' %score)
cmlg
# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_predrfc = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cmrfc = confusion_matrix(y_test, y_predrfc)

score = classifier.score(X_test, y_test)
print('Random Forest model score is %0.4f' %score)
cmrfc
# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_predsvm = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cmkvc = confusion_matrix(y_test, y_predsvm)
score = classifier.score(X_test, y_test)
print('Kernel SVM model score is %0.4f' %score)
cmkvc
# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_predknn = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cmknn = confusion_matrix(y_test, y_predknn)
score = classifier.score(X_test, y_test)
print('KNN model score is %0.4f' %score)
cmknn
# Fitting xgboost

from xgboost import XGBRegressor
xgboost = XGBRegressor()
# Add silent=True to avoid printing out updates with each cycle
xgboost.fit(X_train,y_train,verbose=False)

# Predicting the Test set results
y_predxgboost = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cmxgboost = confusion_matrix(y_test, y_predxgboost)
score = classifier.score(X_test, y_test)
print('XGBoost model score is %0.4f' %score)
cmxgboost