# Import libraries

import numpy as np

import pandas as pd

from time import time

from sklearn.metrics import f1_score



# Read student data

cancer_data = pd.read_csv("../input/data.csv", header=0)

print ("Cancer data read successfully!")

cancer_data.head()
n_cases = cancer_data.shape[0]



n_features = len(list(cancer_data.columns[:-1]))



n_malignant = cancer_data[cancer_data.diagnosis == 'M'].shape[0]



n_benign = cancer_data[cancer_data.diagnosis == 'B'].shape[0]



malignancy_rate = (float(n_malignant)/n_benign) * 100



# Print the results

print ("Total number of cases: {}".format(n_cases))

print ("Number of features: {}".format(n_features))

print ("Number of cases which were positive: {}".format(n_malignant))

print ("Number of students which were negative: {}".format(n_benign))

print ("Malignancy rate in patients: {:.2f}%".format(malignancy_rate))
import matplotlib.pyplot as plt

import seaborn as sns 

%matplotlib inline



# Plotting frequency of cancer types (in our dataset)

sns.countplot(cancer_data['diagnosis'],label="Count")
cancer_data.drop('id',axis=1,inplace=True)

cancer_data.drop('Unnamed: 32',axis=1,inplace=True)

cancer_data['diagnosis'] = cancer_data['diagnosis'].map({'M':1,'B':0})

cancer_data.describe()
# Plotting a correlation graph, to remove any correlated features.

plt.figure(figsize=(14,14))

mean_features = list(cancer_data.columns[1:11])

corr = cancer_data[mean_features].corr()

sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 15},

           xticklabels= mean_features, yticklabels= mean_features,

           cmap= 'coolwarm') # for more on heatmap you can visit Link(http://seaborn.pydata.org/generated/seaborn.heatmap.html)
# Extract feature columns

feature_cols = ['texture_mean','perimeter_mean','smoothness_mean','compactness_mean','symmetry_mean', 'fractal_dimension_mean']



# Extract target column 'diagnosis'

target_col = cancer_data.columns[0] 



# Show the list of columns

print ("Feature columns:\n{}".format(feature_cols))

print ("\nTarget column: {}".format(target_col))



# Separate the data into feature data and target data (X_all and y_all, respectively)

X_all = cancer_data[mean_features]

y_all = cancer_data[target_col]



# Show the feature information by printing the first five rows

print ("\nFeature values:")

print (X_all.head())
from sklearn import cross_validation



# Set the number of training points

num_train = 425



# Set the number of testing points

num_test = X_all.shape[0] - num_train



# Shuffle and split the dataset into the number of training and testing points above

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_all, y_all, test_size=num_test, random_state=10)

X_train_feat = X_train[feature_cols]

X_test_feat = X_test[feature_cols]



# Show the results of the split

print ("Training set has {} samples.".format(X_train.shape[0]))

print ("Testing set has {} samples.".format(X_test.shape[0]))
def train_classifier(clf, X_train, y_train):

    ''' Fits a classifier to the training data. '''

    

    # Start the clock, train the classifier, then stop the clock

    start = time()

    clf.fit(X_train, y_train)

    end = time()

    

    # Print the results

    print ("Trained model in {:.4f} seconds".format(end - start))



    

def predict_labels(clf, features, target):

    ''' Makes predictions using a fit classifier based on F1 score. '''

    

    # Start the clock, make predictions, then stop the clock

    start = time()

    y_pred = clf.predict(features)

    end = time()

    

    # Print and return results

    print ("Made predictions in {:.4f} seconds.".format(end - start))

    return f1_score(target.values, y_pred, pos_label=1)





def train_predict(clf, X_train, y_train, X_test, y_test):

    ''' Train and predict using a classifer based on F1 score. '''

    

    # Indicate the classifier and the training set size

    print ("Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train)))

    

    # Train the classifier

    train_classifier(clf, X_train, y_train)

    

    # Print the results of prediction for both training and testing

    print ("F1 score for training set: {:.4f}.".format(predict_labels(clf, X_train, y_train)))

    print ("F1 score for test set: {:.4f}.".format(predict_labels(clf, X_test, y_test)))
# Import the five supervised learning models from sklearn

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn import svm

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier



# Initialize the three models

clf_A = GaussianNB()

clf_B = DecisionTreeClassifier()

clf_C = svm.SVC()

clf_D = KNeighborsClassifier()

clf_E = RandomForestClassifier()



# Run trainign and prediction for each model

train_predict(clf_A, X_train_feat, y_train, X_test_feat, y_test)

train_predict(clf_B, X_train_feat, y_train, X_test_feat, y_test)

train_predict(clf_C, X_train_feat, y_train, X_test_feat, y_test)

train_predict(clf_D, X_train_feat, y_train, X_test_feat, y_test)

train_predict(clf_E, X_train_feat, y_train, X_test_feat, y_test)
# Will take in the parameters and try all combinations, to find the best parameters for the model.

from sklearn.grid_search import GridSearchCV

from sklearn.metrics import make_scorer

def train_predict_gridsearch(clf, parameters, X_train, y_train, X_test, y_test):

    f1_scorer = make_scorer(f1_score,pos_label=1)



    grid_obj = GridSearchCV(estimator = clf, param_grid = parameters, scoring = f1_scorer)



    grid_obj = grid_obj.fit(X_train, y_train)



    clf = grid_obj.best_estimator_



    # Report the final F1 score for training and testing after parameter tuning

    print ("Tuned model has a training F1 score of {:.4f}.".format(predict_labels(clf, X_train, y_train)))

    print ("Tuned model has a testing F1 score of {:.4f}.".format(predict_labels(clf, X_test, y_test)))

    print ("The best params are: ",grid_obj.best_params_)
# taking parameters for Decison tree Classifier

parameters = {'max_features': ['auto', 'sqrt', 'log2'],

              'min_samples_split': list(range(2, 11)), 

              'min_samples_leaf': list(range(2, 11))}



clf = DecisionTreeClassifier()

train_predict_gridsearch(clf,parameters,X_train,y_train, X_test, y_test)
# taking parameters for SVM

parameters = {'kernel':('linear', 'rbf'),  'gamma':[0.001, 0.01, 0.1, 1]}

clf = svm.SVC()

train_predict_gridsearch(clf,parameters,X_train,y_train, X_test, y_test)
clf = KNeighborsClassifier()

# taking parameters for KNN

parameters = {'n_neighbors': list(range(1, 11)),

              'leaf_size': list(range(1,11)),

              'weights': ['uniform', 'distance']}

train_predict_gridsearch(clf,parameters,X_train,y_train, X_test, y_test)
# Training on all the 10 real valued features

random_forest_classifier = RandomForestClassifier()

train_predict(random_forest_classifier, X_train, y_train, X_test, y_test)
#Create a series with feature importances:

featimp = pd.Series(random_forest_classifier.feature_importances_, index=mean_features).sort_values(ascending=False)

print(featimp)
# Training on the dataset with the 5 important features

imp_features = ['concave points_mean','radius_mean','perimeter_mean','area_mean','concavity_mean']

X_train_imp = X_train[imp_features]

X_test_imp = X_test[imp_features]

random_forest_classifier = RandomForestClassifier()

train_predict(random_forest_classifier, X_train_imp, y_train, X_test_imp, y_test)
# taking parameters for Random Forest classifier with all the 10 real valued features

parameters = {'max_depth': [3, None],

              'max_features': [1,3,10],

              'min_samples_split': [2,5,10],

              'min_samples_leaf': [2,5,10],

              'bootstrap': [True, False],

              'criterion': ['gini', 'entropy']}

clf = RandomForestClassifier()

train_predict_gridsearch(clf,parameters,X_train,y_train, X_test, y_test)
def plot_confusion_matrix(cm, classes,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)

    

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix

import itertools

clf = RandomForestClassifier(bootstrap=True, min_samples_leaf=3, min_samples_split=10, criterion='entropy', max_features=3, max_depth=None)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# save confusion matrix and slice into four pieces

confusion = confusion_matrix(y_test, y_pred)

class_names = [0,1]

# print(confusion)

#[row, column]

TP = confusion[1, 1]

TN = confusion[0, 0]

FP = confusion[0, 1]

FN = confusion[1, 0]



print ("True Positive: ", TP)

print ("True Negative: ", TN)

print ("False Positive: ", FP)

print ("False Negative: ", FN)



plt.figure()

plot_confusion_matrix(confusion, classes=class_names,

                      title='Confusion matrix')
from sklearn.model_selection import KFold

X = np.array(X_all)

y = np.array(y_all)

kf = KFold(n_splits=5)

kf.get_n_splits(X)

print(kf) 

clf = RandomForestClassifier(bootstrap=True, min_samples_leaf=3, min_samples_split=10, criterion='entropy', max_features=3, max_depth=None)

for train_index, test_index in kf.split(X):

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print ("F1 score for test set: {:.4f}.".format(f1_score(y_test, y_pred, pos_label=1)))