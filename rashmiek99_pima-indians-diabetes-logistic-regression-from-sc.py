import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from random import randrange

import warnings

warnings.filterwarnings("ignore")



import os

print(os.listdir("../input"))

diabetes_df = pd.read_csv("../input/diabetes.csv")

diabetes_df.head()
print("How many null values in the dataset?:",diabetes_df.isnull().any().sum())
#Just take the values, ignoring the labels and index

diabetes_df = diabetes_df.values

diabetes_df
X = diabetes_df[:,0:8] #Predictors

y = diabetes_df[:,8] #Target



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)



logistic_model = LogisticRegression(fit_intercept=True,C=1e15)

logistic_model.fit(X_train,y_train)

predicted = logistic_model.predict(X_test)



print("Confusion Matrix")

matrix = confusion_matrix(y_test,predicted)

print(matrix)



print("\nClassification Report")

report = classification_report(y_test,predicted)

print(report)



lr_accuracy = accuracy_score(y_test, predicted)

print('Logistic Regression Accuracy of Scikit Model: {:.2f}%'.format(lr_accuracy*100))
#find the mininum and maximum value of each column

def dataset_minmax(dataset):

    minmax = list()

    

    for i in range(len(dataset[0])):

        col_values = [row[i] for row in dataset]

        

        value_min = min(col_values)

        value_max = max(col_values)



        minmax.append([value_min, value_max])

    

    return minmax



#rescale the value of each column to be within 0 and 1

def normalize_dataset(dataset, minmax):

    for row in dataset:

        for i in range(len(row)):

            row[i]= (row[i]-minmax[i][0]) / (minmax[i][1]-minmax[i][0])
#Predicts an output value for a row given a set of coefficients.



def predict(row, coefficients):

    z = coefficients[0]

    for i in range(len(row)-1):

        z += coefficients[i + 1] * row[i]

    return 1.0 / (1.0 + np.exp(-z))
# Estimate logistic regression coefficients using stochastic gradient descent



def get_coefficients(train, l_rate, n_steps):

    coef = [0.0 for i in range(len(train[0]))]

    

    for step in range(n_steps): #steps times

        sum_error = 0



        for row in train: #all rows

        

            z = predict(row, coef)

            

            error = row[-1] - z #z - row[-1]

            

            coef[0] = coef[0] + l_rate * error * z * (1.0 - z) #b0

            

            for i in range(len(row)-1): #each coefficient (b1,b2,b3....)

                coef[i+1] = coef[i+1]+l_rate*error*z*(1.0-z)*row[i]

                

    return coef
def evaluate_model(test,coef):

    

    predictions = []

    for r in test:

        z = round(predict(r,coef))    

        predictions.append(z)

        

    return(predictions)
def logistic_regression(train,test,l_rate,n_steps):    

    

    #get the coefficients from the training set

    coef = get_coefficients(train,l_rate,n_steps)

    

    #use these to validate against the test set

    predictions = evaluate_model(test,coef)

    

    return(predictions)
# Calculate accuracy percentage

def accuracy_metric(actual, predicted):

    correct = 0

    for i in range(len(actual)):

        if actual[i] == predicted[i]:

            correct += 1

    return correct / float(len(actual))
minmax =dataset_minmax(diabetes_df)

normalize_dataset(diabetes_df, minmax)



l_rate = 0.3

n_steps = 100

n_folds = 3



train_set, test_set = train_test_split(diabetes_df, test_size=0.3)



actual = test_set[:,8]

test_set = test_set[:,0:8]



predicted = logistic_regression(train_set, test_set,l_rate,n_steps)



print("Confusion Matrix")

matrix = confusion_matrix(actual,predicted)

print(matrix)



print("\nClassification Report")

report = classification_report(actual,predicted)

print(report)



scores = accuracy_metric(actual, predicted)

print('Logistic Regression Accuracy Of Our Model: {:.2f}%'.format(scores*100))
