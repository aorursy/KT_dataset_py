# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as plt
from fancyimpute import KNN
%matplotlib inline
%config IPCompleter.greedy = True

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
pwd
df_train = pd.read_csv("../input/loan_train.csv")
df_train.head(10)
df_train['Gender'].value_counts()
pd.crosstab(df_train['Gender'],df_train['Loan_Status'])
df_train.describe()
df_train['ApplicantIncome'].hist()
df_train.Gender.describe()
df_train.count()
df_train.apply(lambda x: sum(x.isnull()),axis = 0)
df_train.loc[df_train['Loan_ID']=='LP001448']
col_name = df_train.columns.values
type(col_name)
col_name
col_name = np.delete(col_name,0)
col_name
col_name = np.delete(col_name,11)
col_name
df_train2 = df_train[col_name]
df_train2.head()
df_train_imp = pd.get_dummies(df_train2,dummy_na=True)
df_train_imp.head()
from fancyimpute import KNN
from fancyimpute import IterativeImputer
imp_cols = df_train_imp.columns.values
imp_cols
df_train_imputed = pd.DataFrame(IterativeImputer(verbose=True).fit_transform(df_train_imp),columns= imp_cols)
df_train_imputed['Loan_Status'] = df_train['Loan_Status']
df_train_imputed.Loan_Status = df_train_imputed.Loan_Status.map(dict(Y=1,N=0))
df_train_imputed.head()
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

predictor_var = df_train_imputed.columns.values
predictor_var = np.delete(predictor_var,26)
predictor_var
type(df_train_imputed[predictor_var])
outcome_var = df_train_imputed.iloc[:,-1:]
#Generic function for making a classification model and accessing performance:
def classification_model(model, data, predictors, outcome):
  #Fit the model:
  model.fit(data[predictors],data[outcome])  
  #Make predictions on training set:    
  predictions = model.predict(data[predictors])  
  #Print accuracy
  accuracy = metrics.accuracy_score(predictions,data[outcome])
  print("Accuracy : %s" % "{0:.3%}".format(accuracy))
  #Perform k-fold cross-validation with 5 folds
  kf = KFold(data.shape[0], n_splits=5)
  error = []
  for train, test in kf:
    # Filter training data
    train_predictors = (data[predictors].iloc[train,:])
    # The target we're using to train the algorithm.
    train_target = data[outcome].iloc[train]
    # Training the algorithm using the predictors and target.
    model.fit(train_predictors, train_target)
    #Record error from each cross-validation run
    error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))    
  print ("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))
  #Fit the model again so that it can be refered outside the function:
  model.fit(data[predictors],data[outcome])
outcome_var = 'Loan_Status'
model = LogisticRegression()
#predictor_var = ['Credit_History']
classification_model(model, df_train_imputed,predictor_var,outcome_var)
outcome_var = 'Loan_Status'
model = RandomForestClassifier(n_estimators=100)
classification_model(model, df_train_imputed,predictor_var,outcome_var)