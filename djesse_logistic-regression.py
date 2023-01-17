import pandas as pd



# Read training data

training_binary = pd.read_csv("../input/training_binary.csv")

y_train = training_binary['cuisine']

x_train = training_binary.drop(['cuisine', 'id'],axis=1)



# Read test data

test_binary = pd.read_csv("../input/test_binary.csv")

y_test = test_binary['cuisine']

x_test = test_binary.drop(['cuisine', 'id'],axis=1)
from sklearn.linear_model import LogisticRegression



# Create logistic classifier using default parameters

clf = LogisticRegression()

clf.fit(x_train, y_train)



# Confusing matrix training data default parameters

training_predictions = clf.predict(x_train)

test_predictions = clf.predict(x_test)
# Performance results using default parameters



from sklearn.metrics import precision_score, recall_score, accuracy_score



print("============ Training data =====================")

print("Accuracy: {}".format(accuracy_score(y_train, training_predictions)))



# Precision

print("Average precision rate: {}".format(precision_score(y_train, training_predictions, average='weighted')))



# Recall

print("Average recall rate: {}".format(recall_score(y_train, training_predictions, average='weighted')))



print("\n============ Test data =====================")

print("Accuracy: {}".format(accuracy_score(y_test, test_predictions)))



# Precision

print("Average precision rate: {}".format(precision_score(y_test, test_predictions, average='weighted')))



# Recall

print("Average recall rate: {}".format(recall_score(y_test, test_predictions, average='weighted')))
from sklearn.metrics import confusion_matrix



# Confusion matrix test set using default parameters

pd.DataFrame(confusion_matrix(y_train,training_predictions,labels=list(set(y_train))),index=list(set(y_train)),columns=list(set(y_train)))
# Confusion matrix test set using default parameters

pd.DataFrame(confusion_matrix(y_test,test_predictions,labels=list(set(y_test))),index=list(set(y_test)),columns=list(set(y_test)))
# Perform gridsearch to optimize C and penalty

from sklearn.model_selection import GridSearchCV

import numpy as np



grid_params = {

    'C': [0.001,0.01,0.1,1,10,100],

    'penalty': ['l2', 'l1']

}



gridsearch = GridSearchCV(LogisticRegression(), grid_params)

gridsearch.fit(x_train, y_train)
# Show gridsearch results

pd.DataFrame(gridsearch.cv_results_)
# Fit logistic regression using optimized parameters

clf = LogisticRegression(C = 1, penalty='l2')

clf.fit(x_train, y_train)



# Confusing matrix training data gridsearch found parameters

training_predictions = clf.predict(x_train)

cm_training = pd.DataFrame(confusion_matrix(y_train,training_predictions,labels=list(set(y_train))),index=list(set(y_train)),columns=list(set(y_train)))



# Confusing matrix test data gridsearch found parameters

test_predictions = clf.predict(x_test)

cm_test = pd.DataFrame(confusion_matrix(y_test,test_predictions,labels=list(set(y_test))),index=list(set(y_test)),columns=list(set(y_test)))
print("==============Training data=======================\n")

print("Accuracy on training set: {}".format(accuracy_score(y_train, training_predictions)))

print("Precision on training set: {}".format(precision_score(y_train, training_predictions, average='weighted')))

print("Recall on training set: {}".format(recall_score(y_train, training_predictions, average='weighted')))



print("\n==============Training data=======================")

print("Accuracy on test set: {}".format(accuracy_score(y_test, test_predictions)))

print("Precision on test set: {}".format(precision_score(y_test, test_predictions, average='weighted')))

print("Recall on test set: {}".format(recall_score(y_test, test_predictions, average='weighted')))
# Training confusion matrix

cm_training
# Test confusion matrix

cm_test