#Load Packages

import pandas as pd 

import numpy as np                     # For mathematical calculations 

import seaborn as sns                  # For data visualization 

import matplotlib.pyplot as plt        # For plotting graphs 

%matplotlib inline 

import warnings                        # To ignore any warnings

warnings.filterwarnings("ignore")
#Read data

train=pd.read_csv('../input/train.txt')

test=pd.read_csv('../input/test.txt')
#Maintain a copy of the data

train_original=train.copy() 

test_original=test.copy()
train.head()
test.head()
train.shape, test.shape
train['Loan_Status'].value_counts().plot.bar()
#Independent Variable (Categorical)



train['Gender'].value_counts(normalize=True).plot.bar(title= 'Gender') 

plt.show()

train['Married'].value_counts(normalize=True).plot.bar(title= 'Married') 

plt.show()

train['Self_Employed'].value_counts(normalize=True).plot.bar(title= 'Self_Employed') 

plt.show()

train['Credit_History'].value_counts(normalize=True).plot.bar(title= 'Credit_History') 

plt.show()
#Independent Variable (Numerical)

sns.distplot(train['ApplicantIncome']); 

plt.show()

train['ApplicantIncome'].plot.box() 

plt.show()
df=train.dropna() 

sns.distplot(df['LoanAmount'])

plt.show()

train['LoanAmount'].plot.box() 

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
train.head()
train=train.drop(['Income_bin', 'Coapplicant_Income_bin', 'LoanAmount_bin', 'Total_Income_bin', 'Total_Income'], axis=1)



train['Dependents'].replace('3+', 3,inplace=True)

test['Dependents'].replace('3+', 3,inplace=True) 

train['Loan_Status'].replace('N', 0,inplace=True)

train['Loan_Status'].replace('Y', 1,inplace=True)
matrix = train.corr() 

f, ax = plt.subplots(figsize=(9, 6)) 

sns.heatmap(matrix, vmax=.8, square=True, cmap="BuPu");
#Missing Values

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
#Building the Model

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



LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,intercept_scaling=1,

                   max_iter=100, multi_class='ovr', n_jobs=1,penalty='l2', random_state=1, 

                   solver='liblinear', tol=0.0001,verbose=0, warm_start=False)
pred_cv = model.predict(x_cv)

logistic_accurcy=accuracy_score(y_cv,pred_cv)

logistic_accurcy
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
#Using log transformation

train['Total_Income_log'] = np.log(train['Total_Income'])

sns.distplot(train['Total_Income_log']); 

test['Total_Income_log'] = np.log(test['Total_Income'])
train['EMI']=train['LoanAmount']/train['Loan_Amount_Term'] 

test['EMI']=test['LoanAmount']/test['Loan_Amount_Term']



sns.distplot(train['EMI']);
#Using log transformation

train['EMI_log'] = np.log(train['EMI'])

sns.distplot(train['EMI_log']); 

test['EMI_log'] = np.log(test['EMI'])
train['Balance Income']=train['Total_Income']-(train['EMI']*1000) # Multiply with 1000 to make the units equal

test['Balance Income']=test['Total_Income']-(test['EMI']*1000)



sns.distplot(train['Balance Income']);
#Drop high correlation variables

train=train.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'], axis=1) 

test=test.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'], axis=1)
X = train.drop('Loan_Status',1) 

y = train.Loan_Status                # Save target variable in separate dataset
#LogisticRegression

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
#Descision tree

from sklearn import tree

i=1 

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

    i+=1 

pred_test = model.predict(test)
#Random Forest

from sklearn.ensemble import RandomForestClassifier

i=1 

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

    i+=1 



pred_test = model.predict(test)
from sklearn.model_selection import GridSearchCV



# Provide range for max_depth from 1 to 20 with an interval of 2 and from 1 to 200 with an interval of 20 for n_estimators 

paramgrid = {'max_depth': list(range(1, 20, 2)), 'n_estimators': list(range(1, 200, 20))}



grid_search=GridSearchCV(RandomForestClassifier(random_state=1),paramgrid)



from sklearn.model_selection import train_test_split 

x_train, x_cv, y_train, y_cv = train_test_split(X,y, test_size =0.3, random_state=1)



# Fit the grid search model 

grid_search.fit(x_train,y_train)



GridSearchCV(cv=None, error_score='raise',       estimator=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',            max_depth=None, max_features='auto', max_leaf_nodes=None,            min_impurity_decrease=0.0, min_impurity_split=None,            min_samples_leaf=1, min_samples_split=2,            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,            oob_score=False, random_state=1, verbose=0, warm_start=False),       

fit_params=None, iid=True, n_jobs=1,       

param_grid={'max_depth': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19], 'n_estimators': [1, 21, 41, 61, 81, 101, 121, 141, 161, 181]},       

pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',       

scoring=None, verbose=0)



# Estimating the optimized value 

grid_search.best_estimator_



RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',            max_depth=3, max_features='auto', max_leaf_nodes=None,            min_impurity_decrease=0.0, min_impurity_split=None,            

min_samples_leaf=1, min_samples_split=2,            

min_weight_fraction_leaf=0.0, n_estimators=41, n_jobs=1,            

oob_score=False, random_state=1, verbose=0, warm_start=False)
i=1 

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

    i+=1 

pred_test = model.predict(test) 

pred2=model.predict_proba(test)[:,1]
importances=pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

importances.plot(kind='barh', figsize=(12,8))