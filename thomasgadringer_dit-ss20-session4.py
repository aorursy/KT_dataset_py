# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#Import the data
dataframe = pd.read_table("/kaggle/input/testdataset29/test7.txt")
dataframe.head(10)
from datetime import datetime
#Change into date format
dataframe["Date123456"] = pd.to_datetime(dataframe['Date123456'], format="%d/%m/%Y")
#Grouping by State and date
group1=dataframe.groupby(["State12", "Date123456"]).sum()
group1.head(10)
state_sum=group1.groupby(["State12"]).sum()
print(state_sum)
#Death rate per state
deaths_per_state=state_sum["D"]/state_sum["C"]
print(deaths_per_state)
#Deaths per state in percent
deaths_per_state_percent=deaths_per_state*100
print(deaths_per_state_percent)
#Histogram of death rate per state
import matplotlib.pyplot as plt
plt.hist(deaths_per_state_percent, bins = 8)
plt.xlabel('Death rate', fontsize=18)
plt.ylabel('Number of states', fontsize=16)
#Histogram of death rate per state
import matplotlib.pyplot as plt
plt.hist(deaths_per_state_percent["Alabama"])
plt.xlabel('Death rate', fontsize=18)
plt.ylabel('Alabama', fontsize=16)
#Histogram of death rate per state
import matplotlib.pyplot as plt
plt.hist(deaths_per_state_percent["Florida"])
plt.xlabel('Death rate', fontsize=18)
plt.ylabel('Florida', fontsize=16)
#Histogram of death rate per state
import matplotlib.pyplot as plt
plt.hist(deaths_per_state_percent["Utah"])
plt.xlabel('Death rate', fontsize=18)
plt.ylabel('Utah', fontsize=16)
#Histogram of death rate per state
import matplotlib.pyplot as plt
plt.hist(deaths_per_state_percent["New Yor"])
plt.xlabel('Death rate', fontsize=18)
plt.ylabel('New York', fontsize=16)
#Import libraries for predicting

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
group2=group1.loc["Alaska"]
print(group2)
def classification_model(model, data, predictors, outcome, kfoldnumber):
    ## fit data
    model.fit(data[predictors], data[outcome])
    ## predict train-data
    predictvalues = model.predict(data[predictors])
    ## accuracy
    accuracy = metrics.accuracy_score(predictvalues, data[outcome])
    print("Accuracy: %s" % "{0:.3%}".format(accuracy))
    ## k-fold cross-validation
    kfold = KFold(n_splits=kfoldnumber)
    error =  []
    ##
    for train, test in kfold.split(data):
        
        ## filter training data
        train_data = data[predictors].iloc[train,:]
        train_target = data[outcome].iloc[train]
        
        ## fit data
        model.fit(train_data, train_target)
        ##
        error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
    ##
    print("Cross Validation Score: %s" % "{0:.3%}".format(np.mean(error)))
    ##
    model.fit(data[predictors], data[outcome])
#Predictive analytics: Accuaracy and Cross Validation Score
outcome_var = "C"
predictor_var = ["D"]
model = LogisticRegression(solver="lbfgs", max_iter=10000)
##
classification_model(model, group2, predictor_var, outcome_var, 5)
import graphviz
from sklearn.tree import export_graphviz
#Predictive Analysis: Accuaracy and Cross Validation Score
model = DecisionTreeClassifier()
outcome_var = "C"
predictor_var = ["D"]
##
classification_model(model, group1, predictor_var, outcome_var, 5)
#Predictive Analysis: With graph
model = DecisionTreeClassifier()
outcome_var = "C"
predictor_var = ["D"]
##
classification_model(model, group1, predictor_var, outcome_var, 5)
dot_data = export_graphviz(model, out_file=None, feature_names=predictor_var, filled=True, rounded=True, special_characters=True)
graph=graphviz.Source(dot_data)
graph
dataframe.describe()