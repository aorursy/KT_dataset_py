# Import libraries

import numpy as np

import pandas as pd

from time import time

from sklearn.metrics import f1_score



# Read cancer data

cancer_data = pd.read_csv("../input/data.csv")



print (cancer_data.head())

print ("Cancer data read successfully!")
cancer_data.drop('id', axis = 1, inplace=True)

cancer_data.drop('Unnamed: 32', axis = 1, inplace=True)

cancer_data['diagnosis'] = cancer_data['diagnosis'].map({'M':1,'B':0})

cancer_data.head()
# Calls image as a static png

%matplotlib inline 

import matplotlib.pyplot as plt



# TODO: Calculate number of subjects

n_subjects = cancer_data.shape[0]



# TODO: Calculate number of features

n_features = cancer_data.shape[1]-1



# TODO: Calculate number of subjects with malignant characteristics

n_M = cancer_data[cancer_data['diagnosis']==1].shape[0]



# TODO: Calculate number of subjects with benign characteristics

n_B = cancer_data[cancer_data['diagnosis']==0].shape[0]



# TODO: Calculate graduation rate

cancer_rate = (float(n_M)/n_subjects)*100



# Print the results

print ("Total number of students: {}".format(n_subjects))

print ("Number of features: {}".format(n_features))

print ("Number of subjects with malignant characteristics: {}".format(n_M))

print ("Number of subjects with benign characteristics: {}".format(n_B))

print ("Cancer rate of the subjects: {:.2f}%".format(cancer_rate))



# Visualize the results

plt.hist(cancer_data['diagnosis'])

plt.title('Diagnosis (M=1 , B=0)')

plt.show()
# Extract feature columns

feature_cols = list(cancer_data.columns[1:])



# Extract target column 'diagnosis'

target_col = cancer_data.columns[0] 



# Show the list of columns

print ("Feature columns:\n{}".format(feature_cols))

print ("\nTarget column: {}".format(target_col))



# Separate the data into feature data and target data (X_all and y_all, respectively)

X_all = cancer_data[feature_cols]

y_all = cancer_data[target_col]



# Show the feature information by printing the first five rows

print ("\nFeature values:")

print (X_all.head())
# TODO: Import any additional functionality you may need here

from sklearn.model_selection import train_test_split



# TODO: Set the number of training points

num_train = 455



# Set the number of testing points

num_test = X_all.shape[0] - num_train



# Shuffle and split the dataset into the number of training and testing points above



X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size = num_test)



# Show the results of the split

print ("Training set has {} samples.".format(X_train.shape[0]))

print ("Testing set has {} samples.".format(X_test.shape[0]))
from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from __future__ import division



def train_classifier(clf, X_train, y_train):

    ''' Fits a classifier to the training data. '''

    

    # Start the clock, train the classifier, then stop the clock

    start = time()

    clf.fit(X_train, y_train)

    end = time()

    

    # Print the results

    print ("Trained model in {:.4f} seconds".format(end - start))



    

def predict_labels(clf, features, target):

    ''' Makes predictions using a fit classifier based on accuracy score. '''

    

    # Start the clock, make predictions, then stop the clock

    start = time()

    y_pred = clf.predict(features)

    end = time()

    

    # Print confusion matrix

    CM = confusion_matrix(target.values, y_pred)

    print (CM)

    

    TN = CM[0][0]

    FN = CM[1][0]

    TP = CM[1][1]

    FP = CM[0][1]

    

    # Print False Positive Rate

    print ("False Positive Rate")

    print (float(FP / (FP+TN)))

    

    # Print True Positive Rate

    print ("True Positive Rate")

    print (float(TP / (TP+FN)))

    

    # Print ROC AUC Score

    print ("ROC AUC Score")

    print (roc_auc_score(target.values, y_pred))

    

    # Print and return results

    print ("Made predictions in {:.4f} seconds.".format(end - start))

    return accuracy_score(target.values, y_pred)



def train_predict(clf, X_train, y_train, X_test, y_test):

    ''' Train and predict using a classifer based on accuracy score. '''

    

    # Indicate the classifier and the training set size

    print ("Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train)))

    

    # Train the classifier

    train_classifier(clf, X_train, y_train)

    

    # Print the results of prediction for both training and testing

    print ("Accuracy score for training set: {:.4f}.".format(predict_labels(clf, X_train, y_train)))

    print ("Accuracy score for test set: {:.4f}.".format(predict_labels(clf, X_test, y_test)))

    print ("")
# Import the two supervised learning models from sklearn

# from sklearn import DecisionTreeClassifier & BaggingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import BaggingClassifier

# from sklearn import RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier

# from sklearn.metrics import roc_auc_score 

from sklearn.metrics import roc_auc_score



# Initialize the three models

clf_A = BaggingClassifier(DecisionTreeClassifier())

clf_B = RandomForestClassifier()



# Set up the training set sizes

X_train_255 = X_train[0:255]

y_train_255 = y_train[0:255]



X_train_355 = X_train[0:355]

y_train_355 = y_train[0:355]



X_train_455 = X_train[0:455]

y_train_455 = y_train[0:455]



# Execute the 'train_predict' function for each classifier and each training set size

# train_predict(clf, X_train, y_train, X_test, y_test)

train_predict(clf_A, X_train_255, y_train_255, X_test, y_test)

train_predict(clf_A, X_train_355, y_train_355, X_test, y_test)

train_predict(clf_A, X_train_455, y_train_455, X_test, y_test)

print ("")

train_predict(clf_B, X_train_255, y_train_255, X_test, y_test)

train_predict(clf_B, X_train_355, y_train_355, X_test, y_test)

train_predict(clf_B, X_train_455, y_train_455, X_test, y_test)