# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the 

#input directory



from subprocess import check_output

print(check_output(["ls", "../input/"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#Reading the Data set 

df = pd.read_csv("../input/WA_Fn-UseC_-HR-Employee-Attrition.csv")
#Quick Data Exploration

#few top rows can be read by using the function head()



df.head(10)
#The summary of numerical fields can be looked by using describe() 



df.describe()
#Check missing values in the dataset

df.apply(lambda x: sum(x.isnull()),axis=0)
#There is no missing value in data
#Let’s analyze Age first. Since the extreme values are practically possible, i.e. some people might 

#leave because of Age. So instead of treating them as outliers, let’s try a log transformation to 

#nullify their effect:

df['DailyRate'].hist(bins=20)
df['Age'].hist(bins=20)
df['Age_log'] = np.log(df['Age'])

df['Age_log'].hist(bins=20)
df['DailyRate_log'] = np.log(df['DailyRate'])

df['DailyRate_log'].hist(bins=20)
#Building a Predictive Model in Python

#Since, sklearn requires all inputs to be numeric, all the categorical variables to be converted 

#into numeric by encoding the categories. 



from sklearn.preprocessing import LabelEncoder



var_mod = ['Attrition', 'BusinessTravel', 'Department', 'Department','EducationField','Gender','JobRole','MaritalStatus','Over18','OverTime']



 

le = LabelEncoder()



for i in var_mod:

    df[i] = le.fit_transform(df[i])

df.dtypes 
#The required module is imported, then we will define a generic #classification function, which takes 

#a model #as input and determines the #Accuracy and Cross-Validation scores.



#Import models from scikit learn module:

from sklearn.linear_model import LogisticRegression

from sklearn.cross_validation import KFold   #For K-fold cross validation

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier, export_graphviz

from sklearn import metrics
##Generic function for making a classification model and accessing performance:

def classification_model(model, data, predictors, outcome):

  #Fit the model:

  model.fit(data[predictors],data[outcome])

  

  #Make predictions on training set:

  predictions = model.predict(data[predictors])

  

  #Print accuracy

  accuracy = metrics.accuracy_score(predictions,data[outcome])

  print ("Accuracy : %s" % "{0:.3%}".format(accuracy))

  #Perform k-fold cross-validation with 5 folds

  kf = KFold(data.shape[0], n_folds=5)

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

    

    result = "{0:.3%}".format(np.mean(error))

    print ("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))





  #Fit the model again so that it can be refered outside the function:

  model.fit(data[predictors],data[outcome]) 

#Logistic Regression



#Let’s make our first Logistic Regression model. One way would be to take #all the variables into the

#model but this might result in overfitting #(don’t worry if you’re unaware of this terminology yet).

#In simple words, #taking all variables might result in the model understanding complex #relations 

#specific to the data and will not generalize well



outcome_var = 'Attrition'

model = LogisticRegression()

predictor_var = ['JobSatisfaction']

classification_model(model, df,predictor_var,outcome_var)

#We can try different combination of variables:



model = LogisticRegression()

predictor_var = ['Age_log','BusinessTravel','DailyRate_log','Department','DistanceFromHome','Education','EducationField','EmployeeCount','EmployeeNumber','EnvironmentSatisfaction','Gender','HourlyRate',	'JobInvolvement','JobLevel','JobRole','JobSatisfaction','MaritalStatus','MonthlyIncome','MonthlyRate','NumCompaniesWorked','Over18','OverTime','PercentSalaryHike','PerformanceRating','RelationshipSatisfaction','StandardHours','StockOptionLevel','TotalWorkingYears','TrainingTimesLastYear','WorkLifeBalance','YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager']

classification_model(model, df,predictor_var,outcome_var)
#Decision Tree



#Decision tree is another method for making a predictive model. It is known #to provide higher accuracy

#than logistic regression model



model = DecisionTreeClassifier()

predictor_var = ['Age_log','BusinessTravel','DailyRate_log','Department','DistanceFromHome','Education','EducationField','EmployeeCount','EmployeeNumber','EnvironmentSatisfaction','Gender','HourlyRate',	'JobInvolvement','JobLevel','JobRole','JobSatisfaction','MaritalStatus','MonthlyIncome','MonthlyRate','NumCompaniesWorked','Over18','OverTime','PercentSalaryHike','PerformanceRating','RelationshipSatisfaction','StandardHours','StockOptionLevel','TotalWorkingYears','TrainingTimesLastYear','WorkLifeBalance','YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager']

classification_model(model, df,predictor_var,outcome_var)
#Random Forest

#Random forest is another algorithm for solving the classification problem.



model = RandomForestClassifier(n_estimators=100)

predictor_var = ['Age_log','BusinessTravel','DailyRate_log','Department','DistanceFromHome','Education','EducationField','EmployeeCount','EmployeeNumber','EnvironmentSatisfaction','Gender','HourlyRate',	'JobInvolvement','JobLevel','JobRole','JobSatisfaction','MaritalStatus','MonthlyIncome','MonthlyRate','NumCompaniesWorked','Over18','OverTime','PercentSalaryHike','PerformanceRating','RelationshipSatisfaction','StandardHours','StockOptionLevel','TotalWorkingYears','TrainingTimesLastYear','WorkLifeBalance','YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager']

classification_model(model, df,predictor_var,outcome_var)
#Create a series with feature importances:

featimp = pd.Series(model.feature_importances_, index=predictor_var).sort_values(ascending=False)

print (featimp)
#Let’s use the top 5 variables for creating a model. Also, we will modify the parameters of random

#forest model a little bit:



model = RandomForestClassifier(n_estimators=25, min_samples_split=25, max_depth=7, max_features=1)

predictor_var = ['MonthlyIncome','HourlyRate','TotalWorkingYears','EmployeeNumber','OverTime']

classification_model(model, df,predictor_var,outcome_var)
#Notice that although accuracy reduced, but the cross-validation score is improving showing that the 

#model is generalizing well. Remember that #random forest models are not exactly repeatable. Different

#runs will result in slight variations because #of randomization. But the output should stay in the

#ballpark.You would have noticed that even after some #basic parameter tuning on random forest, we have

#reached a cross-validation accuracy only slightly better #than the original logistic regression model. 