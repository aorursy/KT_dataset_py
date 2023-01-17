import itertools

import matplotlib.pyplot as plt 

import numpy as np 

import pandas as pd

import scipy
testfile='../input/data_set_ALL_AML_independent.csv'

trainfile='../input/data_set_ALL_AML_train.csv'

patient_cancer='../input/actual.csv'



train = pd.read_csv(trainfile)

test = pd.read_csv(testfile)

patient_cancer = pd.read_csv(patient_cancer)
train.head()
# Remove "call" columns from training a test dataframes

train_keepers = [col for col in train.columns if "call" not in col]

test_keepers = [col for col in test.columns if "call" not in col]



train = train[train_keepers]

test = test[test_keepers]

train.head()
# Transpose the columns and rows so that genes become features and rows become observations

train = train.T

test = test.T

train.head()
# Clean up the column names for training data

train.columns = train.iloc[1]

train = train.drop(["Gene Description", "Gene Accession Number"]).apply(pd.to_numeric)



# Clean up the column names for training data

test.columns = test.iloc[1]

test = test.drop(["Gene Description", "Gene Accession Number"]).apply(pd.to_numeric)



train.head()
# Reset the index. The indexes of two dataframes need to be the same before you combine them

train = train.reset_index(drop=True)



# Subset the first 38 patient's cancer types

pc_train = patient_cancer[patient_cancer.patient <= 38].reset_index(drop=True)



# Combine dataframes for first 38 patients: Patient number + cancer type + gene expression values

train = pd.concat([pc_train,train], axis=1)





# Handle the test data for patients 38 through 72

# Clean up the index

test = test.reset_index(drop=True)



# Subset the last patient's cancer types to test

pc_test = patient_cancer[patient_cancer.patient > 38].reset_index(drop=True)



# Combine dataframes for last patients: Patient number + cancer type + gene expression values

test = pd.concat([pc_test,test], axis=1)

sample = train.iloc[:,2:].sample(n=100, axis=1)

sample["cancer"] = train.cancer

sample.describe().round()
from sklearn import preprocessing
sample = sample.drop("cancer", axis=1)

sample.plot(kind="hist", legend=None, bins=20, color='k')

sample.plot(kind="kde", legend=None);
sample_scaled = pd.DataFrame(preprocessing.scale(sample))

sample_scaled.plot(kind="hist", normed=True, legend=None, bins=10, color='k')

sample_scaled.plot(kind="kde", legend=None);
# StandardScaler to remove mean and scale to unit variance

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(train.iloc[:,2:])

scaled_train = scaler.transform(train.iloc[:,2:])

scaled_test = scaler.transform(test.iloc[:,2:])



x_train = train.iloc[:,2:]

y_train = train.iloc[:,1]

x_test = test.iloc[:,2:]

y_test = test.iloc[:,1]
# Grid Search for tuning parameters

from sklearn.model_selection import GridSearchCV

# RandomizedSearch for tuning (possibly faster than GridSearch)

from sklearn.model_selection import RandomizedSearchCV

# Bayessian optimization supposedly faster than GridSearch

from bayes_opt import BayesianOptimization



# Metrics

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss



## Models

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier
# CHERCHEZ FOR PARAMETERS

def cherchez(estimator, param_grid, search):

    """

    This is a helper function for tuning hyperparameters using teh two search methods.

    Methods must be GridSearchCV or RandomizedSearchCV.

    Inputs:

        estimator: Logistic regression, SVM, KNN, etc

        param_grid: Range of parameters to search

        search: Grid search or Randomized search

    Output:

        Returns the estimator instance, clf

    

    """   

    try:

        if search == "grid":

            clf = GridSearchCV(

                estimator=estimator, 

                param_grid=param_grid, 

                scoring=None,

                n_jobs=-1, 

                cv=10, 

                verbose=0,

                return_train_score=True

            )

        elif search == "random":           

            clf = RandomizedSearchCV(

                estimator=estimator,

                param_distributions=param_grid,

                n_iter=10,

                n_jobs=-1,

                cv=10,

                verbose=0,

                random_state=1,

                return_train_score=True

            )

    except:

        print('Search argument has to be "grid" or "random"')

        sys.exit(0)

        

    # Fit the model

    clf.fit(X=scaled_train, y=y_train)

    

    return clf   
# Function for plotting the confusion matrices

def plot_confusion_matrix(cm, title="Confusion Matrix"):

    """

    Plots the confusion matrix. Modified verison from 

    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    Inputs: 

        cm: confusion matrix

        title: Title of plot

    """

    classes=["AML", "ALL"]    

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.bone)

    plt.title(title)

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=0)

    plt.yticks(tick_marks, classes)

    plt.ylabel('Actual')

    plt.xlabel('Predicted')

    thresh = cm.mean()

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j]), 

                 horizontalalignment="center",

                 color="white" if cm[i, j] < thresh else "black")    
# Logistic Regression

# Paramaters

logreg_params = {} 

logreg_params["C"] =  [0.01, 0.1, 10, 100]

logreg_params["fit_intercept"] =  [True, False]

logreg_params["warm_start"] = [True,False]

logreg_params["random_state"] = [1]



lr_dist = {}

lr_dist["C"] = scipy.stats.expon(scale=.01)

lr_dist["fit_intercept"] =  [True, False]

lr_dist["warm_start"] = [True,False]

lr_dist["random_state"] = [1]



logregression_grid = cherchez(LogisticRegression(), logreg_params, search="grid")

acc = accuracy_score(y_true=y_test, y_pred=logregression_grid.predict(scaled_test))

cfmatrix_grid = confusion_matrix(y_true=y_test, y_pred=logregression_grid.predict(scaled_test))

print("**Grid search results**")

print("Best training accuracy:\t", logregression_grid.best_score_)

print("Test accuracy:\t", acc)



logregression_random = cherchez(LogisticRegression(), lr_dist, search="random")

acc = accuracy_score(y_true=y_test, y_pred=logregression_random.predict(scaled_test))

cfmatrix_rand = confusion_matrix(y_true=y_test, y_pred=logregression_random.predict(scaled_test))

print("**Random search results**")

print("Best training accuracy:\t", logregression_random.best_score_)

print("Test accuracy:\t", acc)



plt.subplots(1,2)

plt.subplots_adjust(left=-0.5, bottom=None, right=None, top=None, wspace=0.5, hspace=None)

plot_confusion_matrix(cfmatrix_rand, title="Random Search Confusion Matrix")

plt.subplot(121)

plot_confusion_matrix(cfmatrix_grid, title="Grid Search Confusion Matrix")
# SVM

svm_param = {

    "C": [.01, .1, 1, 5, 10, 100],

    "gamma": [0, .01, .1, 1, 5, 10, 100],

    "kernel": ["rbf"],

    "random_state": [1]

}



svm_dist = {

    "C": scipy.stats.expon(scale=.01),

    "gamma": scipy.stats.expon(scale=.01),

    "kernel": ["rbf"],

    "random_state": [1]

}



svm_grid = cherchez(SVC(), svm_param, "grid")

acc = accuracy_score(y_true=y_test, y_pred=svm_grid.predict(scaled_test))

cfmatrix_grid = confusion_matrix(y_true=y_test, y_pred=svm_grid.predict(scaled_test))

print("**Grid search results**")

print("Best training accuracy:\t", svm_grid.best_score_)

print("Test accuracy:\t", acc)



svm_random = cherchez(SVC(), svm_dist, "random")

acc = accuracy_score(y_true=y_test, y_pred=svm_random.predict(scaled_test))

cfmatrix_rand = confusion_matrix(y_true=y_test, y_pred=svm_random.predict(scaled_test))

print("**Random search results**")

print("Best training accuracy:\t", svm_random.best_score_)

print("Test accuracy:\t", acc)



plt.subplots(1,2)

plt.subplots_adjust(left=-0.5, bottom=None, right=None, top=None, wspace=0.5, hspace=None)

plot_confusion_matrix(cfmatrix_rand, title="Random Search Confusion Matrix")

plt.subplot(121)

plot_confusion_matrix(cfmatrix_grid, title="Grid Search Confusion Matrix")
# KNN

knn_param = {

    "n_neighbors": [i for i in range(1,30,5)],

    "weights": ["uniform", "distance"],

    "algorithm": ["ball_tree", "kd_tree", "brute"],

    "leaf_size": [1, 10, 30],

    "p": [1,2]

}



knn_dist = {

    "n_neighbors": scipy.stats.randint(1,33),

    "weights": ["uniform", "distance"],

    "algorithm": ["ball_tree", "kd_tree", "brute"],

    "leaf_size": scipy.stats.randint(1,1000),

    "p": [1,2]

}



knn_grid = cherchez(KNeighborsClassifier(), knn_param, "grid")

acc = accuracy_score(y_true=y_test, y_pred=knn_grid.predict(scaled_test))

cfmatrix_grid = confusion_matrix(y_true=y_test, y_pred=svm_grid.predict(scaled_test))

print("**Grid search results**")

print("Best training accuracy:\t", knn_grid.best_score_)

print("Test accuracy:\t", acc)



knn_random = cherchez(KNeighborsClassifier(), knn_dist, "random")

acc = accuracy_score(y_true=y_test, y_pred=knn_random.predict(scaled_test))

cfmatrix_rand = confusion_matrix(y_true=y_test, y_pred=knn_random.predict(scaled_test))

print("**Random search results**")

print("Best training accuracy:\t", knn_random.best_score_)

print("Test accuracy:\t", acc)



plt.subplots(1,2)

plt.subplots_adjust(left=-0.5, bottom=None, right=None, top=None, wspace=0.5, hspace=None)

plot_confusion_matrix(cfmatrix_rand, title="Random Search Confusion Matrix")

plt.subplot(121)

plot_confusion_matrix(cfmatrix_grid, title="Grid Search Confusion Matrix")
# Decision tree classifier

dtc_param = {

    "max_depth": [None],

    "min_samples_split": [2],

    "min_samples_leaf": [1],

    "min_weight_fraction_leaf": [0.],

    "max_features": [None],

    "random_state": [4],

    "max_leaf_nodes": [None], # None = infinity or int

    "presort": [True, False]

}





dtc_grid = cherchez(DecisionTreeClassifier(), dtc_param, "grid")

acc = accuracy_score(y_true=y_test, y_pred=dtc_grid.predict(scaled_test))

cfmatrix_grid = confusion_matrix(y_true=y_test, y_pred=dtc_grid.predict(scaled_test))

print("**Grid search results**")

print("Best training accuracy:\t", dtc_grid.best_score_)

print("Test accuracy:\t", acc)



plot_confusion_matrix(cfmatrix_grid, title="Decision Tree Confusion Matrix")