# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import seaborn as sns                  # For data visualization 
import matplotlib.pyplot as plt        # For plotting graphs 
%matplotlib inline 
import warnings   # To ignore any warnings 
warnings.filterwarnings("ignore")
train_data = pd.read_csv('../input/bank-data/train_data.csv')
test_data = pd.read_csv('../input/bank-data/test_data.csv')
train_org = train_data
test_org = test_data
train_data.head()
train_data.columns
test_data.head()
test_data.columns
train_data.dtypes
train_data.shape,test_data.shape
train_data['Gender'].fillna(train_data['Gender'].mode()[0], inplace=True) 
train_data['Married'].fillna(train_data['Married'].mode()[0], inplace=True) 
train_data['Dependents'].fillna(train_data['Dependents'].mode()[0], inplace=True) 
train_data['Self_Employed'].fillna(train_data['Self_Employed'].mode()[0], inplace=True) 
train_data['Credit_History'].fillna(train_data['Credit_History'].mode()[0], inplace=True)

train_data['Loan_Amount_Term'].fillna(train_data['Loan_Amount_Term'].mode()[0], inplace=True)
train_data['LoanAmount'].fillna(train_data['LoanAmount'].median(), inplace=True)

train_data.isnull().sum()
test_data['Gender'].fillna(test_data['Gender'].mode()[0], inplace=True) 
test_data['Married'].fillna(test_data['Married'].mode()[0], inplace=True) 
test_data['Dependents'].fillna(test_data['Dependents'].mode()[0], inplace=True) 
test_data['Self_Employed'].fillna(test_data['Self_Employed'].mode()[0], inplace=True) 
test_data['Credit_History'].fillna(test_data['Credit_History'].mode()[0], inplace=True)

test_data['Loan_Amount_Term'].fillna(test_data['Loan_Amount_Term'].mode()[0], inplace=True)
test_data['LoanAmount'].fillna(test_data['LoanAmount'].median(), inplace=True)

test_data.isnull().sum()
train_data['Loan_Status'].value_counts(normalize=True)
train_data['Loan_Status'].value_counts().plot.bar(color="c")
plt.figure(1)
plt.subplot(221)
train_data['Gender'].value_counts(normalize=True).plot.bar(figsize=(20,10), title= 'Gender',color="c") 
plt.subplot(222) 
train_data['Married'].value_counts(normalize=True).plot.bar(title= 'Married',color="c") 
plt.subplot(223) 
train_data['Self_Employed'].value_counts(normalize=True).plot.bar(title= 'Self_Employed',color="c") 
plt.subplot(224) 
train_data['Credit_History'].value_counts(normalize=True).plot.bar(title= 'Credit_History',color="c") 
plt.show()

plt.figure(1) 
plt.subplot(131) 
train_data['Dependents'].value_counts(normalize=True).plot.bar(figsize=(24,6), title= 'Dependents',color="c") 
plt.subplot(132) 
train_data['Education'].value_counts(normalize=True).plot.bar(title= 'Education',color="c") 
plt.subplot(133) 
train_data['Property_Area'].value_counts(normalize=True).plot.bar(title= 'Property_Area',color="c") 
plt.show()
plt.figure(1) 
plt.subplot(131) 
sns.distplot(train_data['ApplicantIncome']); 
plt.subplot(132) 
train_data['ApplicantIncome'].plot.box(figsize=(16,5)) 
train_data.boxplot(column='ApplicantIncome', by = 'Education') 
plt.suptitle("")

Gender=pd.crosstab(train_data['Gender'],train_data['Loan_Status']) 
Married=pd.crosstab(train_data['Married'],train_data['Loan_Status']) 
Dependents=pd.crosstab(train_data['Dependents'],train_data['Loan_Status']) 
Education=pd.crosstab(train_data['Education'],train_data['Loan_Status']) 
Self_Employed=pd.crosstab(train_data['Self_Employed'],train_data['Loan_Status']) 
Credit_History=pd.crosstab(train_data['Credit_History'],train_data['Loan_Status']) 
Property_Area=pd.crosstab(train_data['Property_Area'],train_data['Loan_Status']) 

Gender.div(Gender.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(5,5))
plt.show()

Married.div(Married.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(5,5)) 
plt.show() 

Dependents.div(Dependents.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(5,5)) 
plt.show() 

Education.div(Education.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(5,5)) 
plt.show() 

Self_Employed.div(Self_Employed.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(5,5)) 
plt.show()

Credit_History.div(Credit_History.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(5,5)) 
plt.show() 

Property_Area.div(Property_Area.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(5,5)) 
plt.show()

bins_income=[0,2000,5000,8000,81000] 
group=['Low','Average','High', 'Very high'] 
train_data['Income_buckets']=pd.cut(train_data['ApplicantIncome'],bins_income,labels=group)
Income_buckets=pd.crosstab(train_data['Income_buckets'],train_data['Loan_Status']) 
Income_buckets.div(Income_buckets.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 
plt.xlabel('ApplicantIncome') 
P = plt.ylabel('Percentage')

bins_loan_amount=[0,100,200,700] 
group=['Low','Average','High'] 
train_data['LoanAmount_buckets']=pd.cut(train_data['LoanAmount'],bins_loan_amount,labels=group)
LoanAmount_buckets=pd.crosstab(train_data['LoanAmount_buckets'],train_data['Loan_Status']) 
LoanAmount_buckets.div(LoanAmount_buckets.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 
plt.xlabel('LoanAmount') 
P = plt.ylabel('Percentage')
plt.show()
train_data=train_data.drop(['Loan_ID','Customer_ID','Income_buckets', 'LoanAmount_buckets'], axis=1)
test_data=test_data.drop(['Loan_ID','Customer_ID'], axis=1)
train_data['Dependents'].replace('3+', 3,inplace=True) 
test_data['Dependents'].replace('3+', 3,inplace=True) 
train_data['Loan_Status'].replace('N', 0,inplace=True) 
train_data['Loan_Status'].replace('Y', 1,inplace=True)
matrix = train_data.corr() 
f, ax = plt.subplots(figsize=(9, 6)) 
sns.heatmap(matrix, vmax=.8, square=True, cmap="YlOrRd");
x = train_data.drop('Loan_Status',1)
y = train_data.Loan_Status
x=pd.get_dummies(x) 
train_data=pd.get_dummies(train_data) 
test_data=pd.get_dummies(test_data)
from sklearn.linear_model import LogisticRegression as lr 
from sklearn.model_selection import StratifiedKFold as skf
from sklearn.metrics import accuracy_score as acc

i=1 
kfolds = skf(n_splits=10,random_state=1,shuffle=True) 
for train_i,test_i in kfolds.split(x,y):     
    print('\n kfold {} / {}'.format(i,kfolds.n_splits))     
    xtrain,xval = x.loc[train_i],x.loc[test_i]     
    ytrain,yval = y[train_i],y[test_i]         
    model = lr(random_state=1)     
    model.fit(xtrain, ytrain)     
    pred_test = model.predict(xval)     
    score = acc(yval,pred_test)     
    print('Accuracy:',score)     
    i+=1 
prediction_test_data = model.predict(test_data) 
prediction = model.predict_proba(xval)[:,1]
from sklearn import metrics as m
fpr, tpr, _ = m.roc_curve(yval,prediction) 
auc = m.roc_auc_score(yval, prediction) 
plt.figure(figsize=(10,8)) 
plt.plot(fpr,tpr,label="validation, auc="+str(auc)) 
plt.xlabel('False Positive Rate') 
plt.ylabel('True Positive Rate') 
plt.legend(loc=4) 
plt.show()
test_org['Prediction']=prediction_test_data
test_org[['Loan_ID','Customer_ID','Prediction']]
