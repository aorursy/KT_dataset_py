import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings("ignore")



train=pd.read_csv("../input/train_u6lujuX_CVtuZ9i.csv")

test=pd.read_csv("../input/test_Y3wMUE5_7gLdaTN.csv")



train_original=train.copy()

test_original=test.copy()

train.columns
test.columns

train.dtypes

train.shape
test.shape
plt.figure(1)

plt.subplot(121)

df=train.dropna()

sns.distplot(df['LoanAmount']);



plt.subplot(122)

train['LoanAmount'].plot.box(figsize=(16,5))



plt.show()
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
model=LogisticRegression()

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
submission['Loan_Status']=pred_test

submission['Loan_ID']=test_original['Loan_ID']