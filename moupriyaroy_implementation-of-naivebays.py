# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
def fit(X_train, Y_train):

    result = {}

    class_values = set(Y_train)

    for current_class in class_values:

        result[current_class] = {}

        result["total_data"] = len(Y_train)

        current_class_rows = (Y_train == current_class)

        X_train_current = X_train[current_class_rows]

        Y_train_current = Y_train[current_class_rows]

        num_features = X_train.shape[1]

        result[current_class]["total_count"] = len(Y_train_current)

        for j in range(1, num_features + 1):

            result[current_class][j] = {}

            all_possible_values = set(X_train[:, j - 1])

            for current_value in all_possible_values:

                result[current_class][j][current_value] = (X_train_current[:, j - 1] == current_value).sum()

                

    return result
def probability(dictionary, x, current_class):

    output = np.log(dictionary[current_class]["total_count"]) - np.log(dictionary["total_data"])

    num_features = len(dictionary[current_class].keys()) - 1;

    for j in range(1, num_features + 1):

        xj = x[j - 1]

        count_current_class_with_value_xj = dictionary[current_class][j][xj] + 1

        count_current_class = dictionary[current_class]["total_count"] + len(dictionary[current_class][j].keys())

        current_xj_probablity = np.log(count_current_class_with_value_xj) - np.log(count_current_class)

        output = output + current_xj_probablity

    return output
def predictSinglePoint(dictionary, x):

    classes = dictionary.keys()

    best_p = -1000

    best_class = -1

    first_run = True

    for current_class in classes:

        if (current_class == "total_data"):

            continue

        p_current_class = probability(dictionary, x, current_class)

        if (first_run or p_current_class > best_p):

            best_p = p_current_class

            best_class = current_class

        first_run = False

    return best_class
def predict(dictionary, X_test):

    y_pred = []

    for x in X_test:

        x_class = predictSinglePoint(dictionary, x)

        y_pred.append(x_class)

    return y_pred
def makeLabelled(column):

    second_limit = column.mean()

    first_limit = 0.5 * second_limit

    third_limit = 1.5*second_limit

    for i in range (0,len(column)):

        if (column[i] < first_limit):

            column[i] = 0

        elif (column[i] < second_limit):

            column[i] = 1

        elif(column[i] < third_limit):

            column[i] = 2

        else:

            column[i] = 3

    return column
from sklearn import datasets

iris = datasets.load_iris()

X = iris.data

Y = iris.target
for i in range(0,X.shape[-1]):

    X[:,i] = makeLabelled(X[:,i])
from sklearn import model_selection

X_train,X_test,Y_train,Y_test = model_selection.train_test_split(X,Y,test_size=0.25,random_state=0)
dictionary = fit(X_train,Y_train)
Y_pred = predict(dictionary,X_test)
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(Y_test,Y_pred))

print(confusion_matrix(Y_test,Y_pred))