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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV , GridSearchCV
df = pd.read_csv('/kaggle/input/iris-flower-dataset/IRIS.csv')
df.head()
df['species'].value_counts()
# Checking attributes of data and it's dtype
df.info()
# Checking if there is any null value
df.isna().sum()
## now for converting all string values into categorial values
# turn categorial variables into numbers
for label, content in df.items():
    if not pd.api.types.is_numeric_dtype(content):
         df[label] = pd.Categorical(content).codes +1
df.head()
df.info()
df['species'].value_counts()
df.head()
# split data into x and y
x = df.drop('species', axis =1)
y = df.species

x.head()
y.head()
# Split data into train and test
np.random.seed(42)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2)
# put models in a dictionary
clf = { 'logistic regressor' : LogisticRegression(),
         'random forest' : RandomForestClassifier()}

# create a function to fit and score models
def fit_and_score(clf, xtrain, xtest, ytrain, ytest):
    """
    fits and evaluates machine learning models
    """
    # set random seed
    
    np.random.seed(42)
    # make dictionary to store model scores
    model_scores = {}
    # loop through models
    for name, model in clf.items():
        # fit the model to the data
        model.fit(xtrain, ytrain)
        #evaluate the model and append it's score into model_scores
        model_scores[name] = model.score(xtest, ytest)
    return model_scores
model_scores = fit_and_score(clf = clf,
                             xtrain = xtrain,
                             xtest = xtest, 
                             ytrain = ytrain, ytest=ytest)
model_scores
# Logistic regression model
m1 = LogisticRegression()
m1.fit(xtrain, ytrain)
# Random forest model
m2 = RandomForestClassifier()
m2.fit(xtrain, ytrain)
# Use the fitted model to make predictions on the test data and
# save the predictions to a variable called y_preds
y_preds = m1.predict(xtest) # For logistic regression model
# Use the fitted model to make predictions on the test data and
# save the predictions to a variable called y_preds
y_preds = m2.predict(xtest) # For random forest  model
# Logistic Regression
# Evaluate the fitted model on the training set using the score() function
m1.score(xtrain, ytrain)
# logistic regression score on test data
m1.score(xtest, ytest)
# Random forest
# Evaluate the fitted model on the training set using the score() function
m2.score(xtrain, ytrain)
# Random forest score on test data
m1.score(xtest, ytest)
