import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.utils.multiclass import unique_labels

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
#Read in our raw data

raw = pd.read_csv('../input/pimaindian/pima-indians-diabetes.data.csv', header = None)
#Checking data types

raw.dtypes
raw.shape #768 records in 9 columns
raw.isnull().sum() #Checking missing values for columns
#Separate X and y

X = raw[raw.columns[0:8]]

y = raw[raw.columns[8]]
#Now we randomly split our train and test dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#Fit our model using Gaussian naive bayes classifiers

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

gnb_fit = gnb.fit(X_train, y_train)
#Fit the test set

y_pred = gnb_fit.predict(X_test)
print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0],(y_test != y_pred).sum()))
#Now we try to standardize the dataset

from sklearn.preprocessing import StandardScaler



X_train_scaled = StandardScaler().fit_transform(X_train)

X_test_scaled = StandardScaler().fit_transform(X_test)
#Standardized prediction not nessecariliy doing well compared to non-standardized predictions

gnb_fit_scaled = gnb.fit(X_train_scaled, y_train)

y_pred_scaled = gnb_fit_scaled.predict(X_test_scaled)

print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0],(y_test != y_pred_scaled).sum()))
def plot_confusion_matrix(y_true, y_pred, classes,

                          normalize=False,

                          title=None,

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if not title:

        if normalize:

            title = 'Normalized confusion matrix'

        else:

            title = 'Confusion matrix, without normalization'



    # Compute confusion matrix

    cm = confusion_matrix(y_true, y_pred)



    fig, ax = plt.subplots()

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...

    ax.set(xticks=np.arange(cm.shape[1]),

           yticks=np.arange(cm.shape[0]),

           # ... and label them with the respective list entries

           xticklabels=classes, yticklabels=classes,

           title=title,

           ylabel='True label',

           xlabel='Predicted label')



    # Rotate the tick labels and set their alignment.

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",

             rotation_mode="anchor")



    # Loop over data dimensions and create text annotations.

    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            ax.text(j, i, format(cm[i, j], fmt),

                    ha="center", va="center",

                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()

    return ax
class_names = (['Have Diabetes', 'Does Not Have Diabetes'])

plot_confusion_matrix(y_test, y_pred, classes=class_names,

                      title='Confusion matrix, without normalization')



plot_confusion_matrix(y_test, y_pred_scaled, classes=class_names, normalize=True,

                      title='Normalized confusion matrix')
#Now trying stochastic gradient descent using scaled data, and grid search to find the best model

from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import GridSearchCV



params = {

    "loss" : ["hinge", "log", "squared_hinge", "modified_huber"],

    "alpha" : [0.0001, 0.001, 0.01, 0.1],

    "penalty" : ["l2", "l1", "none", "elasticnet"],

}



model = SGDClassifier(max_iter=1000)

clf = GridSearchCV(model, param_grid=params)
clf.fit(X_train_scaled, y_train)

print(clf.best_score_)
print(clf.best_estimator_)
y_sgd_pred = clf.predict(X_test_scaled)

print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0],(y_test != y_sgd_pred).sum()))
#Check the confusion matrix

plot_confusion_matrix(y_test, y_sgd_pred, classes=class_names, normalize=True,

                      title='Normalized confusion matrix for Stochastic Gradient Descent')
#Now trying linear SVM to see how this model perform in our dataset

from sklearn.svm import LinearSVC

params = {

    "loss" : ["hinge", "squared_hinge"],

    "C" : [0.1, 1, 10, 100],

}



model = LinearSVC()

clf = GridSearchCV(model, param_grid=params)
clf.fit(X_train_scaled, y_train)

print(clf.best_score_)
print(clf.best_estimator_)
y_svm_pred = clf.predict(X_test_scaled)

print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0],(y_test != y_svm_pred).sum()))
#Check the confusion matrix

plot_confusion_matrix(y_test, y_svm_pred, classes=class_names, normalize=True,

                      title='Normalized confusion matrix for Support Vector Machine')
from sklearn.svm import SVC

params = {

    "degree" : [3, 5, 10],

    "coef0" : [0.1, 1, 10],

    "C" : [0.1, 1, 10],

}



poly_model = SVC(kernel = "poly")

poly_clf = GridSearchCV(poly_model, param_grid=params)
poly_clf.fit(X_train_scaled, y_train)

print(poly_clf.best_score_)
y_poly_pred = poly_clf.predict(X_test_scaled)

print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0],(y_test != y_poly_pred).sum()))
#Check the confusion matrix

plot_confusion_matrix(y_test, y_poly_pred, classes=class_names, normalize=True,

                      title='Normalized confusion matrix for Support Vector Machine')
#Lastly we try the randomforest classifier

from sklearn.ensemble import RandomForestClassifier



rnf_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=8, n_jobs=-1)

rnf_clf.fit(X_train, y_train)
y_rnf_pred = rnf_clf.predict(X_test)

print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0],(y_test != y_rnf_pred).sum()))
#Check the confusion matrix

plot_confusion_matrix(y_test, y_rnf_pred, classes=class_names, normalize=True,

                      title='Normalized confusion matrix for Random Forest')