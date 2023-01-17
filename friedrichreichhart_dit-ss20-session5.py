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
# load csv file into kaggle notebook, store it in a variable
dataframe = pd.read_csv("/kaggle/input/ditloantrain/dit-loan-train.txt")

#
dataframe.head(10)
dataframe.hist(column="ApplicantIncome", by="Loan_Status", bins=15, figsize=(12,7))
dataframe.describe()
def customized_bin(column, cuttingpoints, custom_labels):
    min_val = column.min()
    max_val = column.max()
    
    breaking_points = [min_val] + cuttingpoints + [max_val]
    print(breaking_points)
    
    colBinned = pd.cut(column, bins=breaking_points, labels=custom_labels, include_lowest=True)
    return colBinned

## call the function ##
cuttingpoints = [90, 150, 190]
custom_labels = ["low", "medium", "high", "very high"]
dataframe["LoanAmountBinned"] = customized_bin(dataframe["LoanAmount"], cuttingpoints, custom_labels)

## see output ##
dataframe.head(10)

print(pd.value_counts(dataframe["LoanAmountBinned"], sort=False))
pd.value_counts(dataframe["Married"])
## replacing information ##
def custom_coding(column, dictionary):
    column_coded = pd.Series(column, copy=True)
    for key, value in dictionary.items():
        column_coded.replace(key, value, inplace=True)
    
    return column_coded

## code LoanStatus - Y > 1, N > 0, yes > 1, Yes > 1, ...
dataframe["Loan_Status_Coded"] = custom_coding(dataframe["Loan_Status"], {"N":0, "Y":1, "No":0, "Yes":1, "no":0, "yes":1})

dataframe.head(10)
    
dataframe.describe()
dataframe['Property_Area'].value_counts()
dataframe["ApplicantIncome"].hist(bins=10)
dataframe["ApplicantIncome"].hist(bins=50)
dataframe.boxplot(column="ApplicantIncome", figsize=(15,8))
dataframe.boxplot(column="ApplicantIncome", by="Education", figsize=(15,8))
dataframe["LoanAmount"].hist(bins=50, figsize=(12,8))
dataframe.boxplot(column="LoanAmount", figsize=(12,8))
dataframe["Credit_History"].value_counts(ascending=True)
dataframe["Credit_History"].value_counts(ascending=True, normalize=True)
dataframe["Property_Area"].value_counts()
dataframe["Loan_Status"].value_counts()
dataframe.pivot_table(values="Loan_Status", index=["Credit_History"], aggfunc=lambda x: x.map({"Y": 1, "N":0}).mean())
dataframe["Credit_History"].value_counts()
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(15,6))

ax1 = fig.add_subplot(2,3,1)
ax1.set_xlabel("Credit_History")
ax1.set_ylabel("Count of Applicants")
ax1.set_title("Applicants by Credit_History")
dataframe["Credit_History"].value_counts().plot(kind="bar")


ax2 = fig.add_subplot(2,3,6)
ax2.set_xlabel("Credit_History")
ax2.set_ylabel("Probability of getting loan")
ax2.set_title("Probability of getting loan by credit history")
dataframe.pivot_table(values="Loan_Status", index=["Credit_History"], aggfunc=lambda x: x.map({"Y": 1, "N":0}).mean()).plot(kind="bar")
dataframe["Credit_History"].value_counts().plot(kind="bar")
dataframe.pivot_table(values="Loan_Status", index=["Credit_History"], aggfunc=lambda x: x.map({"Y": 1, "N":0}).mean()).plot(kind="bar")
temp = pd.crosstab(dataframe["Credit_History"], dataframe["Loan_Status"])
temp.plot(kind="bar", stacked=True, color=["red", "blue"])
temp1 = pd.crosstab([dataframe["Credit_History"], dataframe["Gender"]], dataframe["Loan_Status"])
print(temp1)
temp1.plot(kind="bar", stacked=True, color=["orange", "grey"], grid=True, figsize=(12,6))
temp1 = pd.crosstab(dataframe["Credit_History"], [dataframe["Gender"], dataframe["Loan_Status"]])
print(temp1)
dataframe.apply(lambda x: sum(x.isnull()), axis=0)
dataframe["LoanAmount"].fillna(dataframe["LoanAmount"].mean(), inplace=True)
dataframe.apply(lambda x: sum(x.isnull()), axis=0)
dataframe['Self_Employed'].value_counts()
dataframe['Self_Employed'].value_counts(normalize=True)
dataframe["Self_Employed"].fillna("No",inplace=True)
dataframe.apply(lambda x: sum(x.isnull()), axis=0)
dataframe["LoanAmount"].hist(bins=20)
dataframe["LoanAmount_log"] = np.log(dataframe["LoanAmount"])
dataframe["LoanAmount_log"].hist(bins=20)
dataframe.head(10)
dataframe["TotalIncome"] = dataframe["ApplicantIncome"] + dataframe["CoapplicantIncome"]
dataframe["TotalIncome"].hist(bins=20)
dataframe["TotalIncome_log"] = np.log(dataframe["TotalIncome"])
dataframe["TotalIncome_log"].hist(bins=20)
dataframe.apply(lambda x: sum(x.isnull()),axis=0) 
dataframe["Married"].value_counts(normalize=True)
dataframe["Married"].mode()
dataframe["Married"].fillna(dataframe["Married"].mode()[0], inplace=True)
dataframe.apply(lambda x: sum(x.isnull()),axis=0) 
dataframe["Gender"].fillna(dataframe["Gender"].mode()[0], inplace=True)
dataframe["Dependents"].fillna(dataframe["Dependents"].mode()[0], inplace=True)
dataframe["Loan_Amount_Term"].fillna(dataframe["Loan_Amount_Term"].mode()[0], inplace=True)
dataframe["Credit_History"].fillna(dataframe["Credit_History"].mode()[0], inplace=True)
dataframe.apply(lambda x: sum(x.isnull()),axis=0) 
dataframe.dtypes
dataframe.head(6)
dataframe["Dependents"].value_counts()
from sklearn.preprocessing import LabelEncoder
columns_2_encode = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area", "Loan_Status"]

labelEncoder = LabelEncoder()

for i in columns_2_encode:
    dataframe[i] = labelEncoder.fit_transform(dataframe[i])
dataframe.dtypes
dataframe.head(10)
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
        
    
outcome_var = "Loan_Status"
predictor_var = ["Credit_History"]
model = LogisticRegression(solver="lbfgs")
##
classification_model(model, dataframe, predictor_var, outcome_var, 5)
outcome_var = "Loan_Status"
predictor_var = ["Credit_History"]
model = LogisticRegression(solver="lbfgs")
##
classification_model(model, dataframe, predictor_var, outcome_var, 10)
outcome_var = "Loan_Status"
predictor_var = ["Credit_History", "Education", "Married"]
model = LogisticRegression(solver="lbfgs")
##
classification_model(model, dataframe, predictor_var, outcome_var, 10)
outcome_var = "Loan_Status"
predictor_var = ["Education", "Married"]
model = LogisticRegression(solver="lbfgs")
##
classification_model(model, dataframe, predictor_var, outcome_var, 10)
outcome_var = "Loan_Status"
predictor_var = ["Education", "Married", "Property_Area"]
model = LogisticRegression(solver="lbfgs")
##
classification_model(model, dataframe, predictor_var, outcome_var, 10)
outcome_var = "Loan_Status"
predictor_var = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
model = LogisticRegression(solver="lbfgs")
##
classification_model(model, dataframe, predictor_var, outcome_var, 10)
outcome_var = "Loan_Status"
predictor_var = ["Credit_History", "Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
model = LogisticRegression(solver="lbfgs")
##
classification_model(model, dataframe, predictor_var, outcome_var, 10)
from sklearn.tree import DecisionTreeClassifier
##
model = DecisionTreeClassifier()
outcome_var = "Loan_Status"
predictor_var = ["Credit_History", "Gender", "Married", "Education"]
##
classification_model(model, dataframe, predictor_var, outcome_var, 10)

from sklearn.tree import DecisionTreeClassifier
##
model = DecisionTreeClassifier()
outcome_var = "Loan_Status"
predictor_var = ["Credit_History",  "Loan_Amount_Term", "LoanAmount_log"]
##
classification_model(model, dataframe, predictor_var, outcome_var, 5)

from sklearn.tree import DecisionTreeClassifier
##
model = DecisionTreeClassifier()
outcome_var = "Loan_Status"
predictor_var = ["Loan_Amount_Term", "LoanAmount_log", "Credit_History"]
##
classification_model(model, dataframe, predictor_var, outcome_var, 5)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
outcome_var = "Loan_Status"
predictor_var = ["Loan_Amount_Term", "LoanAmount_log", "Credit_History"]
##
classification_model(model, dataframe, predictor_var, outcome_var, 5)
##
outcome_var = "Loan_Status"
predictor_var = ["Credit_History", "Gender", "Married", "Education"]
##
classification_model(model, dataframe, predictor_var, outcome_var, 10)
model = RandomForestClassifier(n_estimators=100)
outcome_var = "Loan_Status"
predictor_var = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Loan_Amount_Term", "Credit_History", "Property_Area", "LoanAmount_log", "TotalIncome_log"]
##
classification_model(model, dataframe, predictor_var, outcome_var, 10)
feature_importance = pd.Series(model.feature_importances_, index=predictor_var).sort_values(ascending=False)
print(feature_importance)
model = RandomForestClassifier(n_estimators=25, min_samples_split=25, max_depth=7, max_features=1)
outcome_var = "Loan_Status"
predictor_var = ["Credit_History", "TotalIncome_log", "LoanAmount_log"]
##
classification_model(model, dataframe, predictor_var, outcome_var, 5)
model = RandomForestClassifier(n_estimators=25, min_samples_split=25, max_depth=7, max_features=1)
outcome_var = "Loan_Status"
predictor_var = ["Credit_History", "TotalIncome_log", "LoanAmount_log", "Dependents", "Property_Area"]
##
classification_model(model, dataframe, predictor_var, outcome_var, 5)
model = DecisionTreeClassifier()
outcome_var = "Loan_Status"
predictor_var = ["Credit_History", "Gender", "Married", "Education"]
##
classification_model(model, dataframe, predictor_var, outcome_var, 5)
import graphviz
from sklearn.tree import export_graphviz
dot_data = export_graphviz(model, out_file=None, feature_names=predictor_var, filled=True, rounded=True, special_characters=True)
graph=graphviz.Source(dot_data)
graph
model = DecisionTreeClassifier()
outcome_var = "Loan_Status"
predictor_var = ["Credit_History", "Loan_Amount_Term", "LoanAmount_log"]
##
classification_model(model, dataframe, predictor_var, outcome_var, 5)
##
dot_data = export_graphviz(model, out_file=None, feature_names=predictor_var, filled=True, rounded=True, special_characters=True)
graph=graphviz.Source(dot_data)
graph
dataframe.head(10)
dataframe2 = dataframe.iloc[:,1:-5]
dataframe2.head(10)
X, y = dataframe2.iloc[:,:-1], dataframe2.iloc[:,-1]
X.head(3)
y.head(3)
import xgboost as xgb
from sklearn.metrics import mean_squared_error
data_matrix = xgb.DMatrix(data=X, label=y)
print(data_matrix)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
print(y_test)
xg_reg = xgb.XGBRegressor(objective='reg:linear', colsample_bytree=0.5, learning_rate=0.1, max_depth=5, alpha=10, n_estimators=10)

xg_reg.fit(X_train, y_train)
preds = xg_reg.predict(X_test)
print (preds)
import matplotlib.pyplot as plt

xgb.plot_tree(xg_reg, num_trees=0)
plt.rcParams['figure.figsize']=[200,40]
plt.show()