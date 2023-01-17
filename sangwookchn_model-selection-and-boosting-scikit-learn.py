import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('../input/social-network-ads/Social_Network_Ads.csv')
dataset.head()
x = dataset.iloc[:, [2,3]]
y = dataset.iloc[:, 4]

#split test and train set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_train)

#building the model
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, Y_train)

#predicting based on the model
y_pred = classifier.predict(X_test)
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = Y_train, cv = 10)
#cv --> number of parts (folds): 10 is the most common practice
#estimator --> put the trained model.
print(accuracies)
print(accuracies.mean()) #this is a better evaluation of the model.
print(accuracies.std()) #to get a sense of variance and bias
#Applying grid search
from sklearn.model_selection import GridSearchCV
parameters = [{"C": [1, 10, 100, 1000], "kernel": ['linear']}, 
              {"C": [1, 10, 100, 1000], "kernel": ['rbf'], 'gamma': [0.5, 0.1, 0.01, 0.001]}]
#we use this parameters list of dictionaries to test out many combinations of parameters and models.
#This dictionary is tailored to SVM model. For different models, different dictionary keys and values should be used.
#C: penalty parameter that reduces overfitting. Gamma: for optimal kernel

#Use this list to train
grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv = 10, n_jobs = -1)
#scoring: how models are evaluated | cv: apply K-fold cross validation
#n_jobs: how much CPU to use. -1 --> all CPU.

grid_search = grid_search.fit(X_train, Y_train)

#Use attributes of grid_search to get the results
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

print(best_accuracy)
print(best_parameters)
dataset2 = pd.read_csv('../input/bank-customer-churn-modeling/Churn_Modelling.csv')
dataset2.head()
#prepare the dataset
x = dataset2.iloc[:, 3:13]
y = dataset2.iloc[:, 13]

from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()
x.iloc[:, 1] = labelencoder_x.fit_transform(x.iloc[:, 1]) #applying on Geography

#apply encoder on Gender as well
labelencoder_x_2 = LabelEncoder()
x.iloc[:, 2] = labelencoder_x_2.fit_transform(x.iloc[:, 2]) #applying on Gender

from keras.utils import to_categorical
encoded = pd.DataFrame(to_categorical(x.iloc[:, 1]))
#no need to encode Gender, as there are only two categories

x = pd.concat([encoded, x], axis = 1)

#Dropping the existing "geography" category, and one of the onehotcoded columns.

x = x.drop(['Geography', 0], axis = 1)

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# ************* Feature selection is not necessary in xgboost, as it is decision-tree algorithm.
#build the model
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, Y_train)
#predict the results
y_pred = classifier.predict(X_test)
y_pred[:5]
#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)
cm
#K-Fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = Y_train, cv = 10)
print(accuracies.mean())
print(accuracies.std())