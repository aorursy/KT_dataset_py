# Libraries
from statsmodels.discrete.discrete_model import Logit
import h2o
from h2o import H2OFrame
from h2o.tree import H2OTree
from h2o.estimators import H2ORandomForestEstimator
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from scipy.stats import pearsonr
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
%matplotlib inline
# Import data
rdata = pd.read_csv('/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv', na_values = [' '])
dim = rdata.shape

# List of variables in raw data
print(*rdata.columns, sep = '\n')
# Review data
n_row = 10
rdata.head(n_row).T
# Missing data checking
data_missing = rdata.isnull()
print(data_missing.iloc[:, :(dim[1]//2)].describe())
print(data_missing.iloc[:, (dim[1]//2):].describe())
# Drop missing data
rdata.dropna(inplace = True)
# Categorical variables list
list_remove = ['customerID', 'tenure', 'MonthlyCharges', 'TotalCharges']
var_cat = [name for name in rdata.columns if name not in list_remove]

# Categorical data 
data_cat = pd.get_dummies(rdata[var_cat], drop_first = True).astype(int)

# Continuous data 
data_cont = rdata[list_remove[1:]].apply(lambda x: pd.to_numeric(x), axis = 1)

# Training data and scaling
data_scaler = MinMaxScaler()
train_data = pd.concat([data_cont, data_cat], axis = 1) # Training data
# List out original encoding names in each categorical variable
for var_name in var_cat:
    print('\n', var_name, '\n')
    print(rdata[var_name].value_counts())
ncol_train_data, threshold = train_data.shape[1], 0.999
duplicated_columns = set()

# Loop through each pair of variables in design matrix
for i in range(ncol_train_data):
    for j in range(ncol_train_data):
        if i != j:
            corr_coef = pearsonr(train_data.iloc[:, i], train_data.iloc[:, j])
            corr_coef = corr_coef[0]
            if corr_coef > threshold:
                duplicated_columns.add(train_data.columns[i])
                duplicated_columns.add(train_data.columns[j])
                
# List out duplicated column names          
print(duplicated_columns)
# Keep one variable
duplicated_columns = list(duplicated_columns)[1:]
# Removing duplicated variables
train_data.drop(duplicated_columns, axis = 1, inplace = True)

# Training data for inferential modelling
X_inf = train_data.drop('Churn_Yes', axis = 1) 

# Training data for predictive modelling
X, y, cv = data_scaler.fit_transform(X_inf), train_data[['Churn_Yes']], 5 
# Correlation plot
plt.rcParams["figure.figsize"] = (20,20)
plt.matshow(X_inf.corr())
plt.xticks(range(len(X_inf.columns)), X_inf.columns)
plt.yticks(range(len(X_inf.columns)), X_inf.columns)
plt.colorbar()
plt.show()
# Logistic Regression
logistic_model = Logit(y, X_inf)
logistic_fit = logistic_model.fit(method = 'bfgs', maxiter = 1000)
print(logistic_fit.summary())
# H2O initialization
h2o.init()
h2o_train_data = H2OFrame(rdata.iloc[:, 1:])
h2o_x_col, h2o_y_col = h2o_train_data.columns[0:(len(h2o_train_data.columns) - 1)], h2o_train_data.columns[-1]

# Random forest
rf_model = H2ORandomForestEstimator()
rf_model.train(h2o_x_col, h2o_y_col, h2o_train_data)

# Variable Importance
rf_model.varimp_plot()
# Logistic Regression with L1 penalty
def cv_Logistic(X, y, cv, C):
    accuracy_score_list = list()
    for c in C:
        logisticClassifier = LogisticRegression(max_iter = 1000, penalty = 'l1', solver = 'liblinear', C = c)
        average_accuracy = np.mean(cross_val_score(logisticClassifier, X, y, cv = cv))
        accuracy_score_list.append(average_accuracy)
        print('Regularization parameters: ', c, 'Accuracy: ', average_accuracy)
    opt_C = max(C, key = lambda x: x in accuracy_score_list)
    return opt_C

# Compile Model
regularization_parameter = [0.1 * i for i in range(1, 21)]
opt_C = cv_Logistic(X, np.ravel(y), cv, regularization_parameter)
# Random Forest 
def cv_RandomForest(X, y, cv):
    rfClassifier = RandomForestClassifier()
    average_accuracy = np.mean(cross_val_score(rfClassifier, X, y, cv = cv))
    print('Accuracy: ', average_accuracy)
    return average_accuracy

# Compile Model
print(cv_RandomForest(X, np.ravel(y), cv))
# Support vector machine with linear kernel
def cv_SVM(X, y, cv, C):
    accuracy_score_list = list()
    for c in C:
        svmClassifier = SVC(C = c, kernel = 'linear')
        average_accuracy = np.mean(cross_val_score(svmClassifier, X, y, cv = cv))
        accuracy_score_list.append(average_accuracy)
        print('Regularization parameters: ', c, 'Accuracy: ', average_accuracy)
    opt_C = max(C, key = lambda x: x in accuracy_score_list)
    return opt_C

# Compile Model
regularization_parameter = [0.1 * i for i in range(1, 21)]
opt_C = cv_SVM(X, np.ravel(y), cv, regularization_parameter)
# XGBoost 
def cv_XGBoost(X, y, cv):
    xgbClassifier = XGBClassifier(booster = 'gblinear')
    average_accuracy = np.mean(cross_val_score(xgbClassifier, X, y, cv = cv))
    print('Accuracy: ', average_accuracy)
    return average_accuracy

# Compile Model
print(cv_XGBoost(X, np.ravel(y), cv))