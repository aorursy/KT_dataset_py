# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


import seaborn as sns
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, RandomizedLasso)
from scipy import stats as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE, f_regression
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
from scipy.stats import norm, skew #for some statistics

# Any results you write to the current directory are saved as output.
loans = pd.read_csv("../input/LoanStats3a.csv", encoding="utf-8", skiprows=1)
loans.head()
loans.info()
total = loans.isnull().sum().sort_values(ascending=False)
total[total>0]
percent = (loans.isnull().sum()/loans.isnull().count()).sort_values(ascending = False)
pd1 = pd.concat([total,percent],axis =1 ,keys=['Total','Percent'])
pd2 = pd1[pd1['Total']>0]
print(pd2)
loans.dropna(subset=['purpose','total_acc','last_pymnt_d','tax_liens','revol_util','title','last_credit_pull_d','pub_rec_bankruptcies','emp_title'], inplace=True)
loans = loans.loc[:, percent < 0.3]
loans.shape
unique = loans.nunique()
unique = unique[unique.values == 1]
loans.drop(labels = list(unique.index), axis =1, inplace=True)
loans.shape
print(loans.emp_length.unique())
loans.emp_length.fillna('0',inplace=True)
loans.emp_length.replace(['n/a'],'Self-Employed',inplace=True)
print(loans.emp_length.unique())
loans.drop(labels = 'zip_code', axis =1, inplace=True)
loans.info()
loans['int_rate'] = loans['int_rate'].str.replace('%', '')
loans['int_rate'] = loans['int_rate'].astype(float)

loans['revol_util'] = loans['revol_util'].str.replace('%', '')
loans['revol_util'] = loans['revol_util'].astype(float)
(loans.purpose.value_counts()*100)/len(loans)
del_loan_purpose = (loans.purpose.value_counts()*100)/len(loans)
del_loan_purpose = del_loan_purpose[(del_loan_purpose < 0.75) | (del_loan_purpose.index == 'other')]

loans.drop(labels = loans[loans.purpose.isin(del_loan_purpose.index)].index, inplace=True)

print(loans.purpose.unique())
(loans.loan_status.value_counts()*100)/len(loans)
loans.info()
loans['issue_month'],loans['issue_year'] = loans['issue_d'].str.split('-', 1).str
loans[['issue_d','issue_month','issue_year']].head()
q = loans["annual_inc"].quantile(0.995)
loans = loans[loans["annual_inc"] < q]
loans["annual_inc"].describe()
loan_correlation = loans.corr()
loan_correlation
loans['issue_month'] = loans['issue_month'].astype(str)
loans['issue_year'] = loans['issue_year'].astype(str)
del loans['issue_d']
loans['earliest_month'],loans['earliest_year'] = loans['earliest_cr_line'].str.split('-', 1).str
loans['earliest_month'] = loans['earliest_month'].astype(str)
loans['earliest_year'] = loans['earliest_year'].astype(str)
del loans['earliest_cr_line']
loans['last_pymnt_d_month'],loans['last_pymnt_d_year'] = loans['last_pymnt_d'].str.split('-', 1).str
loans['last_pymnt_d_month'] = loans['last_pymnt_d_month'].astype(str)
loans['last_pymnt_d_year'] = loans['last_pymnt_d_year'].astype(str)
del loans['last_pymnt_d']
int_cols = [key for key in dict(loans.dtypes) if dict(loans.dtypes)[key] in ['object']]
int_cols
#encoding categorical variables
from sklearn.preprocessing import LabelEncoder
cols =('grade',
 'sub_grade',
 'emp_title',
 'home_ownership',
 'verification_status',
 'loan_status',
 'purpose',
 'title',
 'addr_state',
 'debt_settlement_flag')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(loans[c].values)) 
    loans[c] = lbl.transform(list(loans[c].values))

# shape        
print('Shape all_data: {}'.format(loans.shape))
loans[int_cols].corr(method='spearman')>0.7
#removing columns with high autocorrelation
del loans['funded_amnt']
del loans['funded_amnt_inv']
del loans['loan_amnt']
del loans['total_pymnt_inv']
del loans['total_rec_prncp']
del loans['total_rec_int']
del loans['collection_recovery_fee']
loans.info()
from sklearn.cross_validation import train_test_split

train,test = train_test_split(loans, train_size=0.8 , random_state=100)
df1 = train.copy()
# First extract the target variable which is our House prices
Y = df1.loan_status.values
# Drop price from the house dataframe and create a matrix out of the house data
X = df1.drop(['loan_status'], axis=1)

# Store the column/feature names into a list "colnames"
int_cols = [key for key in dict(X.dtypes) if dict(X.dtypes)[key] in ['float64', 'int64','uint8','int32']]
len(int_cols)
X_train = train[int_cols]
y_train = train['loan_status']
X_test = test[int_cols]
y_test = test['loan_status']
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
#LogisticRegression

#params = {'C':[1, 10, 50, 100, 500, 1000, 2000], 'tol': [0.001, 0.0001, 0.005]}
log_reg = LogisticRegression(solver='newton-cg',max_iter=300, multi_class='multinomial',n_jobs=-1)
#clf = GridSearchCV(log_reg, params, scoring='log_loss', refit='True', n_jobs=-1, cv=5)
fit = log_reg.fit(X_train, y_train)
scaler = StandardScaler().fit(X_test)
X_test = scaler.transform(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(log_reg.score(X_test, y_test)))
import matplotlib.pyplot as plt
y_pred = log_reg.predict(X_test)
from sklearn.metrics import confusion_matrix
labels = ['Fully paid','Charged off','Fully paid(CP not meet)','Charged off(CP not meet)']
print(labels)
cm = confusion_matrix(y_test,y_pred)

print(cm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels( labels)
ax.set_yticklabels( labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
#LightGBM Model

import lightgbm as lgb
import gc

lgb_params1 = {}
lgb_params1['learning_rate'] = 0.1
lgb_params1['n_estimators'] = 300
lgb_params1['max_bin'] = 10
lgb_params1['subsample'] = 0.7
lgb_params1['subsample_freq'] = 12
lgb_params1['colsample_bytree'] = 0.7   
lgb_params1['min_child_samples'] = 600
lgb_params1['seed'] = 1974

#watchlist = [X_test]
clf = lgb.LGBMClassifier(**lgb_params1)
lgb_fit = clf.fit(X_train,y_train)
y_pred = lgb_fit.predict(X_test)
from sklearn.metrics import confusion_matrix
labels = ['Fully paid','Charged off','Fully paid(CP not meet)','Charged off(CP not meet)']
print(labels)
cm = confusion_matrix(y_test,y_pred)

print(cm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels( labels)
ax.set_yticklabels( labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
print("Accuracy:", accuracy_score(y_test,y_pred)) 
#Random Forest Classifier

from sklearn.metrics import accuracy_score 
from sklearn.ensemble import RandomForestClassifier 

clf = RandomForestClassifier() 
print(clf) 

# fit the model on training data and predict on unseen test data
clf.fit(X_train, y_train) 
preds = clf.predict(X_test) 

# check the accuracy of the predictive model
print("Accuracy:", accuracy_score(y_test,preds)) 
y_pred = clf.predict(X_test)
from sklearn.metrics import confusion_matrix
labels = ['Fully paid','Charged off','Fully paid(CP not meet)','Charged off(CP not meet)']
print(labels)
cm = confusion_matrix(y_test,y_pred)

print(cm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels( labels)
ax.set_yticklabels( labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()