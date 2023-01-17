# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



## further libs for dit session

import matplotlib as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/dit-loan-train.txt")

data.head(10)
data.describe()
data['Property_Area'].value_counts()
data['ApplicantIncome'].hist(bins=10)
data['ApplicantIncome'].hist(bins=50)
data.boxplot(column="ApplicantIncome", figsize=(15,8))
data.boxplot(column="ApplicantIncome", by="Education", figsize=(15,8))
data["LoanAmount"].hist(bins=50, figsize=(12,8))
data.boxplot(column="LoanAmount", figsize=(12,8))
temp1 = data["Credit_History"].value_counts(ascending=True)

print(temp1)
temp1 = data["Credit_History"].value_counts(ascending=True, normalize=True)

print(temp1)
temp2 = data.pivot_table(values="Loan_Status", index=["Credit_History"], aggfunc=lambda x: x.map({"Y": 1, "N": 0}).mean())

print(temp2)
temp1 = data["Credit_History"].value_counts(ascending=True)



import matplotlib.pyplot as plt                                          



fig = plt.figure(figsize=(8,4))

ax1 = fig.add_subplot(121)

ax1.set_xlabel('Credit_History')

ax1.set_ylabel('Count of Applicants')

ax1.set_title("Applicants by Credit_History")

temp1.plot(kind='bar')



ax2 = fig.add_subplot(122)

ax2.set_xlabel('Credit_History')

ax2.set_ylabel('Probability of getting loan')

ax2.set_title("Probability of getting loan by credit history")

temp2.plot(kind='bar')

temp3 = pd.crosstab(data['Credit_History'], data['Loan_Status'])

temp3.plot(kind="bar", stacked=True, color=["red", "blue"])
temp4 = pd.crosstab([data['Credit_History'], data['Gender']], data['Loan_Status'])

print(temp4)

temp4.plot(kind='bar', stacked=True, color=['orange','grey'], grid=False, figsize=(12,6))
temp5 = pd.crosstab([data['Gender'], data['Credit_History']], data['Loan_Status'])

temp5.plot(kind='bar', stacked=True, color=['orange','grey'], grid=False, figsize=(12,6))
data.apply(lambda x: sum(x.isnull()),axis=0) 
data["LoanAmount"].fillna(data["LoanAmount"].mean(), inplace=True)

data.apply(lambda x: sum(x.isnull()),axis=0) 
print(data['Self_Employed'].value_counts())

print(data['Self_Employed'].value_counts(normalize=True))

data['Self_Employed'].fillna('No',inplace=True)
data.apply(lambda x: sum(x.isnull()),axis=0) 
data['LoanAmount'].hist(bins=20)
data["LoanAmount_log"] = np.log(data["LoanAmount"])

data["LoanAmount_log"].hist(bins=20)
data.head(10)
data["TotalIncome"] = data["ApplicantIncome"] + data["CoapplicantIncome"]

data["TotalIncome"].hist(bins=20)
data["TotalIncome_log"] = np.log(data["TotalIncome"])

data["TotalIncome_log"].hist(bins=20)
data.apply(lambda x: sum(x.isnull()),axis=0) 
data['Married'].mode()
data['Gender'].fillna(data['Gender'].mode()[0], inplace=True)

data.apply(lambda x: sum(x.isnull()),axis=0) 
data['Married'].fillna(data['Married'].mode()[0], inplace=True)

data['Dependents'].fillna(data['Dependents'].mode()[0], inplace=True)

data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mode()[0], inplace=True)

data['Credit_History'].fillna(data['Credit_History'].mode()[0], inplace=True)
data.apply(lambda x: sum(x.isnull()),axis=0) 
data.dtypes
data.head(6)
from sklearn.preprocessing import LabelEncoder



var_2_encode = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area", "Loan_Status"]



labelEncoder = LabelEncoder()

for i in var_2_encode:

    data[i] = labelEncoder.fit_transform(data[i])
data.dtypes
data.head(10)
from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier, export_graphviz

from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics

from sklearn.model_selection import KFold
def classification_model(model, data, predictors, outcome):

    ## fit the data

    model.fit(data[predictors], data[outcome])

    ## predicit train-data

    predictvalues = model.predict(data[predictors])

    ## accurancy

    accuracy = metrics.accuracy_score(predictvalues, data[outcome])

    print("Accuracy: %s" % "{0:.3%}".format(accuracy))

    ##

    ## k-fold cross-validation

    kfold = KFold(n_splits=5)

    error = []

    ##

    for train, test in kfold.split(data):

        #print("------- run ------")

        #print("traindata")

        #print(train)

        #print("testdata")

        #print(test)

        ##

        ## filter training data

        train_data = (data[predictors].iloc[train,:])

        train_target = data[outcome].iloc[train]

        ##

        ## fit data

        model.fit(train_data, train_target)

        ##

        error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))

    

    print("Cross Validation Score: %s" % "{0:.3%}".format(np.mean(error)))

    ##

    model.fit(data[predictors], data[outcome])

    

    
outcome_var = "Loan_Status"

predictor_var = ["Credit_History"]

model = LogisticRegression(solver="lbfgs")

##

classification_model(model, data, predictor_var, outcome_var)
outcome_var = "Loan_Status"

predictor_var = ["Credit_History"]

model = LogisticRegression(solver="liblinear")

##

classification_model(model, data, predictor_var, outcome_var)
outcome_var = "Loan_Status"

predictor_var = ["Credit_History"]

model = LogisticRegression(solver="newton-cg")

##

classification_model(model, data, predictor_var, outcome_var)
outcome_var = "Loan_Status"

predictor_var = ["Credit_History", "Education", "Married"]

model = LogisticRegression(solver="lbfgs")

##

classification_model(model, data, predictor_var, outcome_var)
outcome_var = "Loan_Status"

predictor_var = ["Credit_History", "Education", "Married", "Self_Employed", "Property_Area"]

model = LogisticRegression(solver="lbfgs")

##

classification_model(model, data, predictor_var, outcome_var)
model = DecisionTreeClassifier()

outcome_var = "Loan_Status"

predictor_var = ["Credit_History", "Gender", "Married", "Education"]

##

classification_model(model, data, predictor_var, outcome_var)
model = DecisionTreeClassifier()

outcome_var = "Loan_Status"

predictor_var = ["Credit_History", "Loan_Amount_Term", "LoanAmount_log"]

##

classification_model(model, data, predictor_var, outcome_var)
model = RandomForestClassifier(n_estimators=100)

outcome_var = "Loan_Status"

predictor_var = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Loan_Amount_Term", "Credit_History", "Property_Area", "LoanAmount_log", "TotalIncome_log"]

##

classification_model(model, data, predictor_var, outcome_var)
feature_importance = pd.Series(model.feature_importances_, index=predictor_var).sort_values(ascending=False)

print(feature_importance)
model = RandomForestClassifier(n_estimators=25, min_samples_split=25, max_depth=7, max_features=1)

outcome_var = "Loan_Status"

predictor_var = ["Credit_History", "TotalIncome_log", "LoanAmount_log"]

##

classification_model(model, data, predictor_var, outcome_var)
model = RandomForestClassifier(n_estimators=25, min_samples_split=25, max_depth=7, max_features=1)

outcome_var = "Loan_Status"

predictor_var = ["Credit_History", "TotalIncome_log", "LoanAmount_log", "Dependents", "Property_Area"]

##

classification_model(model, data, predictor_var, outcome_var)
model = DecisionTreeClassifier()

outcome_var = "Loan_Status"

predictor_var = ["Credit_History", "Gender", "Married", "Education"]

##

classification_model(model, data, predictor_var, outcome_var)
import graphviz

from sklearn.tree import export_graphviz
dot_data = export_graphviz(model, out_file=None, feature_names=predictor_var, filled=True, rounded=True, special_characters=True)

graph=graphviz.Source(dot_data)

graph
model = DecisionTreeClassifier()

outcome_var = "Loan_Status"

predictor_var = ["Credit_History", "Loan_Amount_Term", "LoanAmount_log"]

##

classification_model(model, data, predictor_var, outcome_var)

##

dot_data = export_graphviz(model, out_file=None, feature_names=predictor_var, filled=True, rounded=True, special_characters=True)

graph=graphviz.Source(dot_data)

graph