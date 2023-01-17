



# Import the required libraries



import sys

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

np.set_printoptions(threshold=sys.maxsize,precision=3)

from IPython.display import display

pd.options.display.max_columns = None

from sklearn.metrics import confusion_matrix as cm

from sklearn.metrics import accuracy_score as accuracy

import warnings



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



# Defining a function which will be used later to analyze the accuracy of the model

def PrintAccuracy(classifier_name, y_test, y_pred):

    print(f'Accuracy using {classifier_name} is :- ', (accuracy(y_test,y_pred)*100))



# Ignore python warnings

warnings.filterwarnings('ignore')



# Import the data set

dataset = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')  # This is Dataset Breast Cancer Wisconsin dataset



# Check if there are any missing values

dataset.isnull().sum()  # Here there are no missing values observed.



# Get the dependent and independent variables

x = dataset.iloc[:,2:32].values

y = dataset.iloc[:,1].values.reshape(-1,1)



# Encode diagnosis column

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

diagnosis_encoder = LabelEncoder()

y = diagnosis_encoder.fit_transform(y)

# diagnosis_encoder = OneHotEncoder()

# y = diagnosis_encoder.fit_transform(y.reshape(-1,1)).toarray() # TODO :- Check if there is any other better way



# Split the data into training and test set

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x, y, test_size = 0.25)



# Fit the decision tree classifier in the data set

from sklearn.tree import DecisionTreeClassifier

tree_classifier = DecisionTreeClassifier()

tree_classifier.fit(x_train,y_train)



# Make predictions

tree_pred = tree_classifier.predict(x_test)

PrintAccuracy(classifier_name= 'Decision Tree', y_pred= tree_pred, y_test= y_test)  # 92.3



# Try fitting Logistic Regression to the data

from sklearn.linear_model import LogisticRegression

logistic_classifier = LogisticRegression()

logistic_classifier.fit(x_train,y_train)



# Make predictions

logistic_pred = logistic_classifier.predict(x_test)

PrintAccuracy(classifier_name= 'Logistic Regression', y_pred= logistic_pred, y_test= y_test) # 95.8



# Fit SVM into the dataset

from sklearn.svm import SVC

svc_classifier = SVC(kernel= 'rbf')

svc_classifier.fit(x_train, y_train)



# Make predictions

svc_pred = svc_classifier.predict(x_test)

PrintAccuracy(classifier_name= 'Support Vector Machine', y_pred= svc_pred, y_test= y_test) # 62.23



# Fit Naive Bayes to the dataset

from sklearn.naive_bayes import GaussianNB

nb_classifier = GaussianNB()

nb_classifier.fit(x_train, y_train)



# Make predictions

nb_pred = nb_classifier.predict(x_test)

PrintAccuracy(classifier_name= 'Naive Bayes', y_pred= nb_pred, y_test= y_test) # 94.405



# Fit KNN to the dataset

from sklearn.neighbors import KNeighborsClassifier

knn_classifier = KNeighborsClassifier(n_neighbors= 5 , metric = 'minkowski', p = 2)

knn_classifier.fit(x_train, y_train)



# Make predictions

knn_pred = knn_classifier.predict(x_test)

PrintAccuracy(classifier_name= 'KNN', y_pred= knn_pred, y_test= y_test) # 92.307



# N- Fold Cross Validations for the different model used



# Utility function

from sklearn.model_selection import cross_val_score

def ApplyCrossValidation(regressor, x_train, y_train):

    accuracies = cross_val_score(estimator = regressor, X = x_train, y = y_train, cv = 10)

    return accuracies.mean()*100



# 1. Decision Tree Classifier

print(f'Average accuracy of Decision Tree Classifier after applying 10- Fold CV is {ApplyCrossValidation(tree_classifier, x_train, y_train)}')

# 91.07203630175837



# 2. Logistic Regression

print(f'Average accuracy of Logistic Regression after applying 10- Fold CV is {ApplyCrossValidation(logistic_classifier, x_train, y_train)}')

# 95.25808281338628



# 3. SVM

print(f'Average accuracy of SVM after applying 10- Fold CV is {ApplyCrossValidation(svc_classifier, x_train, y_train)}')

# 62.91548496880317



# 4. Naive Bayes

print(f'Average accuracy of Naive Bayes after applying 10- Fold CV is {ApplyCrossValidation(nb_classifier, x_train, y_train)}')

# 93.85138967668748



# 5. KNN

print(f'Average accuracy of KNN after applying 10- Fold CV is {ApplyCrossValidation(knn_classifier, x_train, y_train)}')

# 93.85138967668748