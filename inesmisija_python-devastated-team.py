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
dataframe = pd.read_csv ("/kaggle/input/Cleared_data_03.06.2020.csv")
dataframe.head(20)
dataframe.tail(80)
dataframe.describe()
dataframe.loc[(dataframe["name health region"] == "Brasília")]
dataframe.loc[(dataframe["name health region"] == "Brasília")], ["region","date", "new cases", "new death", "recovered"]
dataframe.loc[(dataframe["name health region"] == "Brasília")]
dataframe.loc[(dataframe["name health region"] == "Brasília")], ["region","date", " new cases", " new death", "recovered"]
def num_missing(x):
    return sum(x.isnull())

dataframe.apply(num_missing, axis=0)
dataframe.apply(num_missing, axis=1)
dataframe.head(5)
dataframe["New cases"].value_counts()
dataframe.apply(num_missing, axis=0)
dataframe["cases accumulated"].value_counts()
import matplotlib.pyplot as plt

dataframe.boxplot(column="cases accumulated", figsize=(15,7))
dataframe.boxplot(column="cases accumulated", by="week of epedamic", figsize=(15,7))
dataframe.hist(column="cases accumulated", by="week of epedamic", bins=15, figsize=(15,10))
fig, ax = plt.subplots()
dataframe.hist(column="cases accumulated", by="week of epedamic", bins=20, figsize=(40,30), ax=ax)
fig.savefig("test.png")
import matplotlib.pyplot as plt
dataframe["Region"].hist(bins=20)
fig, ax = plt.subplots()
dataframe.hist(column="Region", by="week of epedamic", bins=5, figsize=(15,5), ax=ax)
fig.savefig("test.png")
dataframe.apply(lambda x: sum(x.isnull()),axis=0)
dataframe.dtypes
dataframe.hist(column="cases accumulated", by="Region", bins=15, figsize=(12,7))
dataframe["cases accumulated"].hist(bins=50)
dataframe.boxplot(column="cases accumulated", by="Region", figsize=(25,12))
dataframe["New cases"].hist(bins=50, figsize=(12,8))
dataframe.boxplot(column="New cases", figsize=(12,8))
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
def classification_model(model, data, predictors, outcome, kfoldnumber):
    ## fit data
    model.fit(data[predictors], data[outcome])
    ## predict train-data
    predictvalues = model.predict(data[predictors])
    ## accuracy
    accuracy = metrics.accuracy_score(predictvalues, data[outcome])
    print("Accuracy: %s" % "{0:.3%}".format(accuracy))
    ##
    ## k-fold cross-validation
    kfold = KFold(n_splits=kfoldnumber)
    error =  []
    ##
    for train, test in kfold.split(data):
        #print("------ run ------")
        #print("traindata")
        #print(train)
        #print("testdata")
        #print(test)
        ##
        ## filter training data
        train_data = data[predictors].iloc[train,:]
        train_target = data[outcome].iloc[train]
        ##
        #print("Traindata")
        #print(train_data)
        #print("TrainTarget")
        #print(train_target)
        ##
        ## fit data
        model.fit(train_data, train_target)
        ##
        error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
    ##
    print("Cross Validation Score: %s" % "{0:.3%}".format(np.mean(error)))
    ##
    model.fit(data[predictors], data[outcome])
outcome_var = "New cases"
predictor_var = ["cases accumulated"]
model = LogisticRegression(solver="lbfgs")
##
classification_model(model, dataframe, predictor_var, outcome_var, 5)
