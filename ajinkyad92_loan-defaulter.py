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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier

from scipy.stats import zscore

from sklearn import metrics

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

import warnings

warnings.filterwarnings("ignore")
loan_train = pd.read_csv("../input/train.csv")

loan_train.head(10)
loan_train.m13.value_counts()
sns.countplot(x='m13',data=loan_train)

plt.show()
loan_train.columns
def basic_details(df):

    b = pd.DataFrame()

    b['Null Values'] = df.isnull().sum()

    b['Data Type'] = df.dtypes

    b['Unique Values'] = df.nunique()

    return b
basic_details(loan_train)
loan_train.shape
loan_train_new = loan_train.copy()
loan_train_new.corr()['m13']
loan_train_new = pd.get_dummies(loan_train_new)
loan_train_new.corr()['m13']
loan_train_new.drop(columns='financial_institution_Anderson-Taylor',inplace=True)

loan_train_new.drop(columns='financial_institution_Browning-Hart',inplace=True)

loan_train_new.drop(columns='financial_institution_Chapman-Mcmahon',inplace=True)

loan_train_new.drop(columns='financial_institution_Cole, Brooks and Vincent',inplace=True)

loan_train_new.drop(columns='financial_institution_Edwards-Hoffman',inplace=True)

loan_train_new.drop(columns='financial_institution_Martinez, Duffy and Bird',inplace=True)

loan_train_new.drop(columns='financial_institution_Miller, Mcclure and Allen',inplace=True)

loan_train_new.drop(columns='financial_institution_Nicholson Group',inplace=True)

loan_train_new.drop(columns='financial_institution_OTHER',inplace=True)

loan_train_new.drop(columns='financial_institution_Richards-Walters',inplace=True)

loan_train_new.drop(columns='financial_institution_Richardson Ltd',inplace=True)

loan_train_new.drop(columns='financial_institution_Romero, Woods and Johnson',inplace=True)

loan_train_new.drop(columns='financial_institution_Sanchez, Hays and Wilkerson',inplace=True)

loan_train_new.drop(columns='financial_institution_Sanchez-Robinson',inplace=True)

loan_train_new.drop(columns='financial_institution_Suarez Inc',inplace=True)

loan_train_new.drop(columns='financial_institution_Swanson, Newton and Miller',inplace=True)

loan_train_new.drop(columns='financial_institution_Taylor, Hunt and Rodriguez',inplace=True)

loan_train_new.drop(columns='financial_institution_Thornton-Davis',inplace=True)

loan_train_new.drop(columns='financial_institution_Turner, Baldwin and Rhodes',inplace=True)

loan_train_new.drop(columns='origination_date_2012-01-01',inplace=True)

loan_train_new.drop(columns='origination_date_2012-02-01',inplace=True)

loan_train_new.drop(columns='origination_date_2012-03-01',inplace=True)

loan_train_new.drop(columns='first_payment_date_02/2012',inplace=True)

loan_train_new.drop(columns='first_payment_date_03/2012',inplace=True)

loan_train_new.drop(columns='first_payment_date_04/2012',inplace=True)

loan_train_new.drop(columns='first_payment_date_05/2012',inplace=True)

loan_train_new.drop(columns='loan_id',inplace=True)

# loan_train_new.drop(columns='insurance_percent',inplace=True)

# loan_train_new.drop(columns='insurance_type',inplace=True)

# loan_train_new.drop(columns='unpaid_principal_bal',inplace=True)

# loan_train_new.drop(columns='loan_term',inplace=True)

# loan_train_new.drop(columns='loan_to_value',inplace=True)

# loan_train_new.drop(columns='debt_to_income_ratio',inplace=True)
loan_train_new['Average_credit_score'] = (loan_train_new['borrower_credit_score']+loan_train_new['co-borrower_credit_score'])/ loan_train_new['number_of_borrowers']

loan_train_new.drop(columns='number_of_borrowers',inplace=True)

loan_train_new.drop(columns='borrower_credit_score',inplace=True)

loan_train_new.drop(columns='co-borrower_credit_score',inplace=True)

loan_train_new.head()
loan_train_new['loan_term_years'] = loan_train_new['loan_term']/12

loan_train_new.drop(columns='loan_term',inplace=True)

loan_train_new.head()
loan_train_new['monthly_installment'] = ((loan_train_new.unpaid_principal_bal)*(1+(loan_train_new.interest_rate)/100))/((loan_train_new.loan_term_years)*12)

loan_train_new.head()
cor=loan_train_new.corr()
plt.figure(figsize=(20,15))

sns.heatmap(cor,annot=True)

plt.show()
data = loan_train_new.copy()
x = data.drop(columns='m13')

y = data['m13']
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3 , random_state = 45)
lr = LogisticRegression()

lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)
from sklearn.metrics import accuracy_score , confusion_matrix ,recall_score ,precision_score,f1_score

accuracy_score(y_test, y_pred)
confusion_matrix(y_test, y_pred)
recall_score(y_test, y_pred)
precision_score(y_test, y_pred)
f1_score(y_test,y_pred)
from imblearn.over_sampling import SMOTE
y_train.value_counts()
smt = SMOTE()

x_train, y_train = smt.fit_sample(x_train, y_train)

np.bincount(y_train)
lr = LogisticRegression()

lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)
accuracy_score(y_test, y_pred)
confusion_matrix(y_test, y_pred)
precision_score(y_test, y_pred)
recall_score(y_test, y_pred)
f1_score(y_test,y_pred)
from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = 0.3 , random_state = 45)
from imblearn.under_sampling import NearMiss
nr = NearMiss()

xtrain, ytrain = nr.fit_sample(xtrain, ytrain)

np.bincount(ytrain)
lr = LogisticRegression()

lr.fit(xtrain, ytrain)

y_pred = lr.predict(xtest)
accuracy_score(ytest, y_pred)
confusion_matrix(ytest, y_pred)
precision_score(ytest, y_pred)
recall_score(ytest, y_pred)
f1_score(ytest,y_pred)
from sklearn.model_selection import RandomizedSearchCV

param_grid = {

    'max_depth': range(5, 25, 5),

    'min_samples_leaf': range(50, 150, 50),

    'min_samples_split': range(50, 150, 50),

    'criterion': ["entropy", "gini"],

    'random_state':range(0,101)

}



n_folds = 10



# Instantiate the grid search model

dtree = DecisionTreeClassifier()

random_cv = RandomizedSearchCV(estimator=dtree,

                               param_distributions=param_grid,

                               cv=n_folds, n_iter=25, 

                               scoring = 'neg_mean_absolute_error',

                               n_jobs = -1, verbose = 1, 

                               return_train_score = True,

                               random_state=42)



# Fit the grid search to the data

random_cv.fit(x_train,y_train)
print("best accuracy", random_cv.best_score_)

print(random_cv.best_estimator_)
dtc = random_cv.best_estimator_

dtc.fit(x_train, y_train)

pred_dtc = dtc.predict(x_test)
accuracy_score(y_test, pred_dtc)
confusion_matrix(y_test, pred_dtc)
print('Precision is: ',precision_score(y_test, pred_dtc))

print('Recall is: ',recall_score(y_test, pred_dtc))

print('F1 Score is: ',f1_score(y_test,pred_dtc))
param_grid = {'n_estimators': range(10, 50, 10),

                'max_depth': range(5, 25, 5),

    'min_samples_leaf': range(50, 150, 50),

    'min_samples_split': range(50, 150, 50),

    'criterion': ["entropy", "gini"]

             }



n_folds = 10



# Instantiate the grid search model

rfc = RandomForestClassifier()

random_cv = RandomizedSearchCV(estimator=rfc,

                               param_distributions=param_grid,

                               cv=n_folds, n_iter=25, 

                               scoring = 'neg_mean_absolute_error',

                               n_jobs = -1, verbose = 1, 

                               return_train_score = True,

                               random_state=42)



# Fit the grid search to the data

random_cv.fit(x_train,y_train)
print("best accuracy", random_cv.best_score_)

print(random_cv.best_estimator_)
rfc = random_cv.best_estimator_

rfc.fit(x_train, y_train)
pred_rfc = rfc.predict(x_test)
accuracy_score(y_test, pred_rfc)
confusion_matrix(y_test, pred_rfc)
print('Precision is: ',precision_score(y_test, pred_rfc))

print('Recall is: ',recall_score(y_test, pred_rfc))

print('F1 Score is: ',f1_score(y_test,pred_rfc))
from lightgbm import LGBMClassifier

lightgbm = LGBMClassifier(n_jobs=-1)



lightgbm.fit(x_train,y_train)
pred_lgbm = lightgbm.predict(x_test)
accuracy_score(y_test, pred_lgbm)
confusion_matrix(y_test, pred_lgbm)
print('Precision is: ',precision_score(y_test, pred_lgbm))

print('Recall is: ',recall_score(y_test, pred_lgbm))

print('F1 Score is: ',f1_score(y_test,pred_lgbm))
from xgboost import XGBClassifier

param_grid = {'n_estimators': range(10, 50, 10),

                'max_depth': range(5, 25, 5),

             }



n_folds = 10



# Instantiate the grid search model

xgb = XGBClassifier()

random_cv = RandomizedSearchCV(estimator=xgb,

                               param_distributions=param_grid,

                               cv=n_folds, n_iter=25, 

                               scoring = 'neg_mean_absolute_error',

                               n_jobs = -1, verbose = 1, 

                               return_train_score = True,

                               random_state=42)



# Fit the grid search to the data

random_cv.fit(x_train,y_train)
print("best accuracy", random_cv.best_score_)

print(random_cv.best_estimator_)
xgb = random_cv.best_estimator_