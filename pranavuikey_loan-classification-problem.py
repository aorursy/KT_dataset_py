# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import seaborn as sns                  # For data visualization 

import matplotlib.pyplot as plt        # For plotting graphs 

%matplotlib inline 

import warnings   # To ignore any warnings 

warnings.filterwarnings("ignore")
train = pd.read_csv('../input/bank-loan2/madfhantr.csv')

test = pd.read_csv('../input/bank-loan2/madhante.csv')
train.head()
test.head()
train_original=train.copy() 

test_original=test.copy()
train.columns
test.columns
# Print data types for each variable 

train.dtypes
train.shape,test.shape
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

Gender.div(Gender.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
Married=pd.crosstab(train['Married'],train['Loan_Status']) 

Dependents=pd.crosstab(train['Dependents'],train['Loan_Status']) 

Education=pd.crosstab(train['Education'],train['Loan_Status']) 

Self_Employed=pd.crosstab(train['Self_Employed'],train['Loan_Status']) 

Married.div(Married.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4)) 

plt.show() 

Dependents.div(Dependents.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 

plt.show() 

Education.div(Education.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4)) 

plt.show() 

Self_Employed.div(Self_Employed.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4)) 

plt.show()
Credit_History=pd.crosstab(train['Credit_History'],train['Loan_Status']) 

Property_Area=pd.crosstab(train['Property_Area'],train['Loan_Status']) 

Credit_History.div(Credit_History.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4)) 

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

f, ax = plt.subplots(figsize=(9, 6)) 

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
train=train.drop('Loan_ID',axis=1) 

test=test.drop('Loan_ID',axis=1)
X = train.drop('Loan_Status',1)

y = train.Loan_Status
X=pd.get_dummies(X) 

train=pd.get_dummies(train) 

test=pd.get_dummies(test)
from sklearn.model_selection import train_test_split

x_train, x_cv, y_train, y_cv = train_test_split(X,y, test_size =0.3)
from sklearn.linear_model import LogisticRegression 

from sklearn.metrics import accuracy_score

model = LogisticRegression() 

model.fit(x_train, y_train)
pred_cv = model.predict(x_cv)

accuracy_score(y_cv,pred_cv)
pred_test = model.predict(test)
from sklearn.model_selection import StratifiedKFold
i=1 

kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True) 

for train_index,test_index in kf.split(X,y):     

    print('\n{} of kfold {}'.format(i,kf.n_splits))     

    xtr,xvl = X.loc[train_index],X.loc[test_index]     

    ytr,yvl = y[train_index],y[test_index]         

    model = LogisticRegression(random_state=1)     

    model.fit(xtr, ytr)     

    pred_test = model.predict(xvl)     

    score = accuracy_score(yvl,pred_test)     

    print('accuracy_score',score)     

    i+=1 

pred_test = model.predict(test) 

pred=model.predict_proba(xvl)[:,1]
from sklearn import metrics 

fpr, tpr, _ = metrics.roc_curve(yvl,  pred) 

auc = metrics.roc_auc_score(yvl, pred) 

plt.figure(figsize=(12,8)) 

plt.plot(fpr,tpr,label="validation, auc="+str(auc)) 

plt.xlabel('False Positive Rate') 

plt.ylabel('True Positive Rate') 

plt.legend(loc=4) 

plt.show()
train['Total_Income']=train['ApplicantIncome']+train['CoapplicantIncome'] 

test['Total_Income']=test['ApplicantIncome']+test['CoapplicantIncome']
sns.distplot(train['Total_Income']);
train['Total_Income_log'] = np.log(train['Total_Income']) 

sns.distplot(train['Total_Income_log']); 

test['Total_Income_log'] = np.log(test['Total_Income'])
train['EMI']=train['LoanAmount']/train['Loan_Amount_Term'] 

test['EMI']=test['LoanAmount']/test['Loan_Amount_Term']
sns.distplot(train['EMI']);
train['Balance Income']=train['Total_Income']-(train['EMI']*1000) # Multiply with 1000 to make the units equal test['Balance Income']=test['Total_Income']-(test['EMI']*1000)

test['Balance Income']=test['Total_Income']-(test['EMI']*1000)

sns.distplot(train['Balance Income']);

train=train.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'], axis=1) 

test=test.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'], axis=1)
X = train.drop('Loan_Status',1) 

y = train.Loan_Status
from statistics import mean
i=1 

score_list = []

kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True) 

for train_index,test_index in kf.split(X,y):     

    print('\n{} of kfold {}'.format(i,kf.n_splits))     

    xtr,xvl = X.loc[train_index],X.loc[test_index]     

    ytr,yvl = y[train_index],y[test_index]         



    model = LogisticRegression(random_state=1)     

    model.fit(xtr, ytr)     

    pred_test = model.predict(xvl)     

    score = accuracy_score(yvl,pred_test)     

    print('accuracy_score',score)     

    score_list.append(score)

    i+=1 

pred_test = model.predict(test) 

pred=model.predict_proba(xvl)[:,1]



log_reg = round(model.score(xtr,ytr) * 100, 2)

cv1 = round(mean(score_list)*100,2)



print()

print('Training Score',log_reg)

print('Mean Cross-validation Score:', cv1)
from sklearn import tree
i=1 

score_list = []

kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True) 

for train_index,test_index in kf.split(X,y):     

    print('\n{} of kfold {}'.format(i,kf.n_splits))     

    xtr,xvl = X.loc[train_index],X.loc[test_index]     

    ytr,yvl = y[train_index],y[test_index]         

    model = tree.DecisionTreeClassifier(random_state=1)     

    model.fit(xtr, ytr)     

    pred_test = model.predict(xvl)     

    score = accuracy_score(yvl,pred_test)     

    print('accuracy_score',score)     

    score_list.append(score)

    i+=1 

pred_test = model.predict(test)



d_tree = round(model.score(xtr,ytr) * 100, 2)

cv2 = round(mean(score_list)*100,2)



print()

print('Training Score',d_tree)

print('Mean Cross-validation Score:', cv2)
from sklearn.ensemble import RandomForestClassifier
i=1 

score_list =[]

kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True) 

for train_index,test_index in kf.split(X,y):     

    print('\n{} of kfold {}'.format(i,kf.n_splits))     

    xtr,xvl = X.loc[train_index],X.loc[test_index]     

    ytr,yvl = y[train_index],y[test_index]         

    model = RandomForestClassifier(random_state=1, max_depth=10)     

    model.fit(xtr, ytr)     

    pred_test = model.predict(xvl)     

    score = accuracy_score(yvl,pred_test)     

    print('accuracy_score',score)     

    score_list.append(score)

    i+=1 



pred_test = model.predict(test)

r_forest = round(model.score(xtr,ytr) * 100, 2)

cv3 = round(mean(score_list)*100,2)

print()

print('Training Score',r_forest)

print('Mean Cross-validation Score:', cv3)
from sklearn.model_selection import GridSearchCV
# Provide range for max_depth from 1 to 20 with an interval of 2 and from 1 to 200 with an interval of 20 for n_estimators 

paramgrid = {'max_depth': list(range(1, 20, 2)), 'n_estimators': list(range(1, 200, 20))}
grid_search=GridSearchCV(RandomForestClassifier(random_state=1),paramgrid)
from sklearn.model_selection import train_test_split 

x_train, x_cv, y_train, y_cv = train_test_split(X,y, test_size =0.3, random_state=1)
# Fit the grid search model 

grid_search.fit(x_train,y_train)
# Estimating the optimized value 

grid_search.best_estimator_
i=1 

score_list=[]

kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True) 

for train_index,test_index in kf.split(X,y):     

    print('\n{} of kfold {}'.format(i,kf.n_splits))     

    xtr,xvl = X.loc[train_index],X.loc[test_index]     

    ytr,yvl = y[train_index],y[test_index]         

    model = RandomForestClassifier(random_state=1, max_depth=3, n_estimators=41)     

    model.fit(xtr, ytr)     

    pred_test = model.predict(xvl)     

    score = accuracy_score(yvl,pred_test)     

    print('accuracy_score',score)     

    score_list.append(score)

    i+=1 

pred_test = model.predict(test) 

pred2=model.predict_proba(test)[:,1]

r_forest_fine_tuned = round(model.score(xtr,ytr) * 100, 2)

cv4 = round(mean(score_list)*100,2)

print()

print('Training Score: ',r_forest_fine_tuned)

print('Mean Cross-validation Score: ', cv4)
from xgboost import XGBClassifier
i=1

score_list=[]

kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True) 

for train_index,test_index in kf.split(X,y):     

    print('\n{} of kfold {}'.format(i,kf.n_splits))     

    xtr,xvl = X.loc[train_index],X.loc[test_index]     

    ytr,yvl = y[train_index],y[test_index]         

    model = XGBClassifier(n_estimators=50, max_depth=4)     

    model.fit(xtr, ytr)     

    pred_test = model.predict(xvl)     

    score = accuracy_score(yvl,pred_test)     

    print('accuracy_score',score)

    score_list.append(score)

    i+=1 

pred_test = model.predict(test) 

pred3=model.predict_proba(test)[:,1]

xg_boost = round(model.score(xtr,ytr) * 100, 2)

cv5 = round(mean(score_list)*100,2)

print()

print('Training Score',xg_boost)

print('Mean Cross-validation Score:', cv5)
score_table = pd.DataFrame([log_reg,d_tree,r_forest,r_forest_fine_tuned,xg_boost],columns=['Training Score'])

score_table['Mean Cross Validation Score'] = [cv1,cv2,cv3,cv4,cv5]

score_table['Algorithms'] = ['Logistic Regression','Decision Tree','Random Forest','Random Forest Fine tuned','XG Boost']

score_table.set_index('Algorithms')