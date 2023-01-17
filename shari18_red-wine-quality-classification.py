import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
dataset = pd.read_csv('../input/winequality-red.csv')
dataset.head()
# converting the response varaible to binary classification
bins = [2,6.5,8]
labels=['bad','good'] # '0' for bad and '1' for good
dataset['quality'] = pd.cut(dataset['quality'], bins = bins , labels = labels)

dataset['quality'].value_counts()
label_encoder = LabelEncoder()
# converting the response variable to binary 
dataset['quality'] = label_encoder.fit_transform(dataset['quality'])
dataset['quality'].value_counts()
# separate the dataset as independent and dependent varaibles
X = dataset.drop('quality',axis = 1).values
y = dataset['quality'].values
# create test and training set data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# applying feature scaling to get optimized result
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
# Fitting SVM to training set
# let's keep the kernel to 'linear'
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train,y_train)
# predicting the test set result
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
cm
print(classification_report(y_test, y_pred))
# in the above SVC model always predicts the quality of wine as 'bad',
# so we need to increase the accuarcy of the model
# finding the best model for SVC using Grid Search
parameters = [{'C' : [1.1,1.2,1.5,1.7,1.8,1.9], 'kernel' : ['linear']},
              {'C' : [1.1,1.2,1.5,1.7,1.8,1.9], 'kernel' : ['rbf'], 'gamma':[1.0,1.1,1.2,1.3,1.4,1.5,1.6]}]
grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv = 10, n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
print(best_accuracy)
best_parameters = grid_search.best_params_
print(best_parameters)
# run SVC again with best parameters
classifier_2 = SVC(kernel = 'rbf', gamma = 1.4, C = 1.8)
classifier_2.fit(X_train,y_train)
y_pred = classifier_2.predict(X_test)
print(classification_report(y_test, y_pred))
