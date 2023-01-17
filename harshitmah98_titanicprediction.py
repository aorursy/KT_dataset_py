# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Changing the directory
import os
os.chdir('../input')

# Importing the dataset
dataset = pd.read_csv('train.csv')
X = dataset.iloc[:,[2,4,5,6,7,9,11]].values
y = dataset.iloc[:,1].values
                                            
# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 2:3])
X[:, 2:3] = imputer.transform(X[:, 2:3])
# Dealing with nan values in Port of embarktion
for i in range(0, len(X[:,6])):
    if X[i,6] not in ['C', 'S', 'Q']:
        X[i,6] = 'Q'

# Encoding the data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
X[:, 6] = labelencoder_X.fit_transform(X[:, 6])
onehotencoder = OneHotEncoder(categorical_features = [6])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
# Applying kernel SVM
from sklearn.svm import SVC
classifier = SVC(C = 1, kernel = 'rbf', gamma = 0.1)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.05]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("The best accuracy on training data with K-SVM is: ", end='')
print(best_accuracy)
print("The best parameters on training data with K-SVM is: ", end='')
print(best_parameters)
# Validating on actual test set
dataset_test = pd.read_csv('test.csv')
X_real_test = dataset_test.iloc[:,[1,3,4,5,6,8,10]].values
y_real_test = pd.read_csv('gender_submission.csv').iloc[:, 1].values

# Preprocessing
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X_real_test[:, 2:3])
X_real_test[:, 2:3] = imputer.transform(X_real_test[:, 2:3])
for i in range(0, len(X_real_test[:,6])):
    if X_real_test[i,6] not in ['C', 'S', 'Q']:
        X_real_test[i,6] = 'Q'

imputer = imputer.fit(X_real_test[:,[5]])
X_real_test[:, [5]] = imputer.transform(X_real_test[:, [5]])
labelencoder_X = LabelEncoder()
X_real_test[:, 1] = labelencoder_X.fit_transform(X_real_test[:, 1])
X_real_test[:, 6] = labelencoder_X.fit_transform(X_real_test[:, 6])
onehotencoder = OneHotEncoder(categorical_features = [6])
X_real_test = onehotencoder.fit_transform(X_real_test).toarray()
X_real_test = X_real_test[:, 1:]
sc_X = StandardScaler()
X_real_test = sc_X.fit_transform(X_real_test)
y_actual_pred = classifier.predict(X_real_test)
cm = confusion_matrix(y_real_test, y_actual_pred)
accuracy = (cm[0][0] + cm[1][1])/418
print("Accuracy with Kernel SVM: ", end = "")
print(accuracy)
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_actual_pred = classifier.predict(X_real_test)
cm = confusion_matrix(y_real_test, y_actual_pred)
accuracy = (cm[0][0] + cm[1][1])/418
print("Accuracy with Naive bayes: ", end = "")
print(accuracy)
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 50, criterion = 'entropy')
classifier.fit(X_train, y_train)
y_actual_pred = classifier.predict(X_real_test)
cm = confusion_matrix(y_real_test, y_actual_pred)
accuracy = (cm[0][0] + cm[1][1])/418
print("Accuracy with Random forest: ", end = "")
print(accuracy)