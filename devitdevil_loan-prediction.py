# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd 

import numpy as np                     # For mathematical calculations 

import seaborn as sns                  # For data visualization 

import matplotlib.pyplot as plt        # For plotting graphs 

%matplotlib inline 


train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.columns
test.columns
# Print data types for each variable

train.dtypes
#Let’s look at the shape of the dataset.

train.shape
test.shape
#Frequency table of a variable will give us the count of each category in that variable.

train['Loan_Status'].value_counts()
# Normalize can be set to True to print proportions instead of number

train['Loan_Status'].value_counts(normalize=True)
train['Loan_Status'].value_counts().plot.bar()
plt.figure(1)

plt.subplot(221)

train['Gender'].value_counts(normalize=True).plot.bar(figsize=(20,10), title= 'Gender') 

plt.subplot(222)

train['Married'].value_counts(normalize=True).plot.bar(title= 'Married') 

plt.subplot(223)

train['Self_Employed'].value_counts(normalize=True).plot.bar(title= 'Self_Employed') 

plt.subplot(224)

train['Credit_History'].value_counts(normalize=True).plot.bar(title= 'Credit_History') 

plt.show()

plt.figure(1) 

plt.subplot(131)

train['Dependents'].value_counts(normalize=True).plot.bar(figsize=(24,6), title= 'Dependents') 

plt.subplot(132)

train['Education'].value_counts(normalize=True).plot.bar(title= 'Education') 

plt.subplot(133)

train['Property_Area'].value_counts(normalize=True).plot.bar(title= 'Property_Area') 

plt.show()

plt.figure(1)

plt.subplot(121)

sns.distplot(train['ApplicantIncome']); 

plt.subplot(122)

train['ApplicantIncome'].plot.box(figsize=(16,5)) 

plt.show()

train.boxplot(column='ApplicantIncome', by = 'Education')

plt.suptitle("")

plt.figure(1)

plt.subplot(121) 

sns.distplot(train['CoapplicantIncome']); 

plt.subplot(122)

train['CoapplicantIncome'].plot.box(figsize=(16,5)) 

plt.show()

plt.figure(1)

plt.subplot(121)

df=train.dropna()

sns.distplot(df['LoanAmount']);

plt.subplot(122)

train['LoanAmount'].plot.box(figsize=(16,5)) 

plt.show()

Gender=pd.crosstab(train['Gender'],train['Loan_Status'])

Gender.div(Gender.sum(1).astype(float), axis=0).plot(kind="bar",stacked=True, figsize=(5,5))

Married=pd.crosstab(train['Married'],train['Loan_Status']) 

Dependents=pd.crosstab(train['Dependents'],train['Loan_Status'])

Education=pd.crosstab(train['Education'],train['Loan_Status']) 

Self_Employed=pd.crosstab(train['Self_Employed'],train['Loan_Status']) 

Married.div(Married.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(5,5)) 

plt.show() 

Dependents.div(Dependents.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)

plt.show() 

Education.div(Education.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(5,5))

plt.show() 

Self_Employed.div(Self_Employed.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(5,5)) 

plt.show()

Credit_History=pd.crosstab(train['Credit_History'],train['Loan_Status'])

Property_Area=pd.crosstab(train['Property_Area'],train['Loan_Status']) 

Credit_History.div(Credit_History.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(5,5))

plt.show() 

Property_Area.div(Property_Area.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)

plt.show()
train.groupby('Loan_Status')['ApplicantIncome'].mean().plot.bar()
bins=[0,2500,4000,6000,81000]

group=['Low','Average','High', 'Very high'] 

train['Income_bin']=pd.cut(df['ApplicantIncome'],bins,labels=group)

Income_bin=pd.crosstab(train['Income_bin'],train['Loan_Status'])

Income_bin.div(Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)

plt.xlabel('ApplicantIncome')

P = plt.ylabel('Percentage')

bins=[0,1000,3000,42000]

group=['Low','Average','High']

train['Coapplicant_Income_bin']=pd.cut(df['CoapplicantIncome'],bins,labels=group)

Coapplicant_Income_bin=pd.crosstab(train['Coapplicant_Income_bin'],train['Loan_Status'])

Coapplicant_Income_bin.div(Coapplicant_Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)

plt.xlabel('CoapplicantIncome')

P = plt.ylabel('Percentage')

train['Total_Income']=train['ApplicantIncome']+train['CoapplicantIncome']

bins=[0,2500,4000,6000,81000]

group=['Low','Average','High', 'Very high']

train['Total_Income_bin']=pd.cut(train['Total_Income'],bins,labels=group)

Total_Income_bin=pd.crosstab(train['Total_Income_bin'],train['Loan_Status'])

Total_Income_bin.div(Total_Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)

plt.xlabel('Total_Income')

P = plt.ylabel('Percentage')
bins=[0,100,200,700]

group=['Low','Average','High']

train['LoanAmount_bin']=pd.cut(df['LoanAmount'],bins,labels=group)

LoanAmount_bin=pd.crosstab(train['LoanAmount_bin'],train['Loan_Status'])

LoanAmount_bin.div(LoanAmount_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)

plt.xlabel('LoanAmount')

P = plt.ylabel('Percentage')

train=train.drop(['Income_bin', 'Coapplicant_Income_bin', 'LoanAmount_bin', 'Total_Income_bin', 'Total_Income'], axis=1)

train['Dependents'].replace('3+', 3,inplace=True)

test['Dependents'].replace('3+', 3,inplace=True)

train['Loan_Status'].replace('N', 0,inplace=True)

train['Loan_Status'].replace('Y', 1,inplace=True)

matrix = train.corr()

f,ax = plt.subplots(figsize=(9, 6))

sns.heatmap(matrix, vmax=.8, square=True, cmap="BuPu");
train.isnull().sum()
train['Gender'].fillna(train['Gender'].mode()[0], inplace=True)

train['Married'].fillna(train['Married'].mode()[0], inplace=True)

train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)

train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)

train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)


train['Loan_Amount_Term'].value_counts()
train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)

train['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)

train.isnull().sum()

test['Gender'].fillna(train['Gender'].mode()[0], inplace=True)

test['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)

test['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)

test['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)

test['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)

test['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)

train['LoanAmount_log'] = np.log(train['LoanAmount'])

train['LoanAmount_log'].hist(bins=20)

test['LoanAmount_log'] = np.log(test['LoanAmount'])

train['Total_Income']=train['ApplicantIncome']+train['CoapplicantIncome']

test['Total_Income']=test['ApplicantIncome']+test['CoapplicantIncome']
#Let’s check the distribution of Total Income.



sns.distplot(train['Total_Income']);


train['Total_Income_log'] = np.log(train['Total_Income'])

sns.distplot(train['Total_Income_log']); 

test['Total_Income_log'] = np.log(test['Total_Income'])
train['EMI']=train['LoanAmount']/train['Loan_Amount_Term']

test['EMI']=test['LoanAmount']/test['Loan_Amount_Term']
#Let’s check the distribution of EMI variable.



sns.distplot(train['EMI']);
train['Balance Income']=train['Total_Income']-(train['EMI']*1000) # Multiply with 1000 to make the units equal

test['Balance Income']=test['Total_Income']-(test['EMI']*1000)

sns.distplot(train['Balance Income']);
train=train.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'], axis=1) 

test=test.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'], axis=1)
X = train.drop('Loan_Status',1) 

y= train.Loan_Status
X=pd.get_dummies(X) 

train=pd.get_dummies(train) 

test=pd.get_dummies(test)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,

                                                    train_size=0.8,

                                                    random_state=42)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import  StratifiedKFold

from sklearn.model_selection import cross_val_score

sk_fold = StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
clf = LogisticRegression()

scoring = 'accuracy'

score = cross_val_score(clf, X_train, y_train, cv=sk_fold, n_jobs=1, scoring=scoring)

print(score)
round(np.mean(score)*100, 2)
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()

scoring = 'accuracy'

score = cross_val_score(clf, X_train, y_train, cv=sk_fold, n_jobs=1, scoring=scoring)

print(score)
round(np.mean(score)*100, 2)
from sklearn.ensemble import RandomForestClassifier
clf =RandomForestClassifier()

scoring = 'accuracy'

score = cross_val_score(clf, X_train, y_train, cv=sk_fold, n_jobs=1, scoring=scoring)

print(score)
round(np.mean(score)*100, 2)
from xgboost import XGBClassifier
clf =XGBClassifier()

scoring = 'accuracy'

score = cross_val_score(clf, X_train, y_train, cv=sk_fold, n_jobs=1, scoring=scoring)

print(score)
round(np.mean(score)*100, 2)
from sklearn.model_selection import train_test_split, GridSearchCV 
param_grid={'max_depth': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19], 'n_estimators': [1, 21, 41, 61, 81, 101, 121, 141, 161, 181]},   

grid = GridSearchCV(XGBClassifier(),param_grid,verbose=3)

grid = GridSearchCV(XGBClassifier(),param_grid,verbose=3)
grid.fit(X_train, y_train)
grid.best_params_ 
grid.best_estimator_
clf =XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,

              colsample_bynode=1, colsample_bytree=1, gamma=0,

              learning_rate=0.1, max_delta_step=0, max_depth=1,

              min_child_weight=1, missing=None, n_estimators=81, n_jobs=1,

              nthread=None, objective='binary:logistic', random_state=0,

              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,

              silent=None, subsample=1, verbosity=1)

scoring = 'accuracy'

score = cross_val_score(clf, X_train, y_train, cv=sk_fold, n_jobs=1, scoring=scoring)

print(score)
round(np.mean(score)*100, 2)