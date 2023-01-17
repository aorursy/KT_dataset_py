# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas.api.types import CategoricalDtype

import sklearn

import matplotlib.pyplot as plt

from sklearn import svm

from sklearn.metrics import make_scorer

from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import make_classification

from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn import tree

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.datasets import make_moons, make_circles, make_classification

from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.gaussian_process.kernels import RBF

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import roc_auc_score

import csv

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
'''

Since there is a non utf-8 letter in the test and train data file, it doesn't work 

to simply read the csv in a pandas datafile. Therefore, we need to read the lines separately 

while ignoring the error and concanate a panda data frame step by step.

'''



# First for the training data file

path = '../input/TrainData.csv'

    

with open(path, 'r', encoding='utf-8', errors='ignore') as infile:

     inputs = csv.reader(infile)

     df_train = pd.DataFrame()

        

     for index, row in enumerate(inputs):

        # If the header is read

        if index == 0:

            columns = row[0].split(';')

        # If the values are read, split them at semicolon and conatenate the dataframes

        else:

            values = row[0].split(';')

            df_dummy = pd.DataFrame(values, columns).T

            df_train = pd.concat([df_train, df_dummy] , ignore_index=True)



# then for the test data file

path = '../input/TestData.csv'



with open(path, 'r', encoding='utf-8', errors='ignore') as infile:

     inputs = csv.reader(infile)

     df_test = pd.DataFrame()

        

     for index, row in enumerate(inputs):

        # If the header is read

        if index == 0:

            columns = row[0].split(';')

        # If the values are read, split them at semicolon and conatenate the dataframes

        else:

            values = row[0].split(';')

            df_dummy = pd.DataFrame(values, columns).T

            df_test = pd.concat([df_test, df_dummy] , ignore_index=True)
features = df_train.columns

n_features = len(features)

n_variables = len(df_train[features[0]])
num_features = ['Stammnummer','Tag','Dauer','Anruf-ID','Alter','Kontostand','Anzahl der Ansprachen',

                'Tage seit letzter Kampagne','Anzahl Kontakte letzte Kampagne']

str_features = ['Zielvariable','Monat','Geschlecht','Art der Anstellung','Familienstand',

                'Schulabschlu','Ausfall Kredit','Haus','Kredit','Kontaktart','Ergebnis letzte Kampagne']
# One hot coding

'''

Change objects to category and introduce hot keys.



The data strucure includes variables of type int64, float64 or object. The object type is in this 

case given as string-variables. Since we can't learn from string-values, we'll

change all 'objects' to a 'category'. This also allows us to introduce an order of the data, for example

jan < feb < mar < ... etc.

'''

# copy the original data frames

df_num_train = df_train.copy()

df_num_test = df_test.copy()



# in 'Tage seit letzter Kampagne' change NaN to 1Mio

df_num_train.replace('', 1000000, inplace=True)

df_num_test.replace('', 1000000, inplace=True)



# store numerical values as float variables

for feature in num_features:

    df_num_train[feature] = df_num_train[feature].astype(float)

    df_num_test[feature] = df_num_test[feature].astype(float)



# store string values as hot keys and delete string rows. therefore transform 'object'

# to 'category'

for feature in str_features:

    # if there are only two possibilities (as 'No' and 'Yes') simply 

    # replace them with 0 and 1 

    if len(df_num_train[feature].unique())==2:

        df_num_train[feature] = df_num_train[feature].factorize()[0]

        df_num_test[feature] = df_num_test[feature].factorize()[0]

    # otherwise use one hot coding 

    else:

        df_num_train[feature] = df_train[feature].astype('category')

        df_dummies = pd.get_dummies(df_train[feature], prefix = feature)

        df_num_train = pd.concat([df_num_train, df_dummies], axis=1)

        df_num_test[feature] = df_test[feature].astype('category') 

        df_dummies = pd.get_dummies(df_test[feature], prefix = feature)

        df_num_test = pd.concat([df_num_test, df_dummies], axis=1)

        df_num_train = df_num_train.drop(feature, axis=1)

        df_num_test = df_num_test.drop(feature, axis=1)





df_num_train = df_num_train.drop('Anruf-ID', axis=1)

df_num_test = df_num_test.drop('Anruf-ID', axis=1)



features = df_num_train.columns

n_features = len(features)

n_variables = len(df_num_train[features[0]])
'''

Define 

        X and y, those are used to choose the optimal parameters for 

                 the varying prediction models, and 

        X_val, y_val, those are used to validate the so chosen model. 

'''

y = df_num_train['Zielvariable']

X = df_num_train.drop('Zielvariable', axis=1)

X_val = df_num_test.drop('Zielvariable', axis=1)
'''

To choose optimal parameters, we use Exhaustive Grid Search. To capture 

the unbalanced structure of the distribution 

of 'Zielvariable', we define a score function based on the ROC AUC.

''' 

# Define score-function based on ROC AUC

roc_auc_scorer = make_scorer(roc_auc_score, greater_is_better=True,

                             needs_threshold=True)



# 1. Logistic Regression

model = LogisticRegression()

param_grid = [

  {'C': [1, 10, 100, 1000] , 'class_weight':['balanced', None], 'solver':['liblinear'], 'penalty': ['l1']},

  {'C': [1, 10, 100, 1000], 'class_weight':['balanced', None], 'solver': ['liblinear','newton-cg','sag','lbfgs'], 'penalty': ['l2']},

 ]



# 2. K-Neighbors Classifier

model = KNeighborsClassifier()

param_grid = [

    {'n_neighbors': [30,40,50,60,70] ,'p': [1]}

 ]



# 3. Decision Tree Classifier

model = DecisionTreeClassifier()

param_grid = [

    {'max_depth': [2,3,4,5,6,7,8,9,10]}

 ]



# 4. Random Forest Classifier

model = RandomForestClassifier()

param_grid = [

    {'max_depth': [5,10,15,20,25], 

     'n_estimators': [5,10,15,20]}

]







clf = GridSearchCV(model, param_grid, cv=5, scoring=roc_auc_scorer)

clf.fit(X,y)                
'''

The best parameters for the above models are explicitely given below. 

So lets predict values for the test set.

'''



names = ['Logistic Regression', 'K Nearest Neighbors', 'Decision Tree Classifier', 'Random Forest Classifier']



models = [LogisticRegression(class_weight='balanced', penalty='l1', C=1000),

          KNeighborsClassifier(n_neighbors=40, p=1),

          DecisionTreeClassifier(max_depth=6),

          RandomForestClassifier(max_depth=15, n_estimators=20)]



y_predictions = []



for i,model in enumerate(models):

    print(names[i])

    clfs = model.fit(X,y)

    y_predictions.append(model.predict(X_val))

    