from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity='all'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
%matplotlib inline
import seaborn as sns
train_file="../input/train_AV3.csv"
training_set=pd.read_csv(train_file)
test_file="../input/test_AV3.csv"
test_set=pd.read_csv(test_file)
training_set.shape
test_set.shape
data=pd.concat([training_set,test_set],sort=False)
data.shape
data.isnull().sum().sort_values()
data['Married'].value_counts()
data['Loan_Amount_Term'].value_counts()
data['Gender'].value_counts()
data['Dependents']=data['Dependents'].str.rstrip('+')
data['Dependents'].value_counts()
data['Self_Employed'].value_counts()
data['Credit_History']=data['Credit_History'].astype('object')
data['Credit_History'].value_counts()
data1=data.iloc[:,0:11]
data1.shape
data1.head(10)
Null_check= pd.DataFrame(data1.isnull().sum())  
Null_check=Null_check.reset_index()
Null_check.rename(columns={'index':'feature',0:'cnt'},inplace=True)
def handling_miss(dataset,data):
    for i in np.array(Null_check[Null_check['cnt']>0].index):
        if((data[Null_check['feature'][i]]).dtypes == 'O'):
            data[Null_check['feature'][i]]=data[Null_check['feature'][i]].fillna(data[Null_check.iloc[i,][0]].describe()[2])
        else:
            data[Null_check['feature'][i]]=data[Null_check['feature'][i]].fillna(data[Null_check.iloc[i,][0]].describe()[1])

handling_miss(Null_check,data)

training_set['Credit_History']=training_set['Credit_History'].astype('object')
training_set.isnull().sum().sort_values()
train_null_check=training_set.isnull().sum()
handling_miss(train_null_check,training_set)
test_set['Credit_History']=test_set['Credit_History'].astype('object')
test_data_null=test_set.isnull().sum().sort_values()
handling_miss(test_data_null,test_set)
training_set['TotalIncome']=training_set['ApplicantIncome']+training_set['CoapplicantIncome']
test_set['TotalIncome']=test_set['ApplicantIncome']+test_set['CoapplicantIncome']
test_set['TotalIncome']=test_set['TotalIncome'].astype('float64')

training_set['LoanAmount']=round(training_set['LoanAmount'],1)
training_set.dtypes
training_set.isnull().sum().sort_values()
training_set.info()
training_set.describe()
test_set.info()
plt.figure(figsize=(10,8))
plt.subplot(2,2,1)
sns.distplot(training_set.ApplicantIncome,bins=50,hist=True) 
plt.subplot(2,2,2)
sns.distplot(training_set.CoapplicantIncome,bins=50,hist=True) 
plt.tight_layout()
plt.subplot(2,2,3)
training_set['LoanAmount'].hist(bins=50)
plt.show();

#from the distribution of features below we can see that "ApplicantIncome" and "CoapplicantIncome" are 
#right skewed and also have outliers present in them.

del training_set['ApplicantIncome']
del training_set['CoapplicantIncome']
del test_set['ApplicantIncome']
del test_set['CoapplicantIncome']
training_set=training_set[['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education','Self_Employed', 'LoanAmount', 'Loan_Amount_Term', 
'Credit_History','Property_Area','TotalIncome','Loan_Status']]
credit_check=pd.crosstab(columns=training_set['Loan_Status'],values=training_set['Credit_History'],
           index=training_set['Credit_History'],aggfunc='count')

credit_check
credit_check.plot(kind='bar',stacked=True)

training_set['Education'].value_counts()
Edu_check=pd.crosstab(columns=training_set['Loan_Status'],values=training_set['Education'],index=training_set['Education'],aggfunc='count').apply(lambda r: r/r.sum(), axis=1)
Edu_check
Edu_check.plot(kind='bar')
social_status=pd.crosstab(columns=training_set['Loan_Status'],index=[training_set['Married'],training_set['Dependents']],values=training_set['Loan_Status'],aggfunc='count').apply(lambda x: x/x.sum(),axis=1)
social_status
social_status.plot(kind='bar')
social_status1=pd.crosstab(columns=training_set['Loan_Status'],index=[training_set['Married'],training_set['Dependents']],values=training_set['Loan_Status'],aggfunc='count')
social_status1.plot(kind='bar')
training_set['Self_Employed'].value_counts()
training_set.groupby(['Self_Employed','Loan_Status'])['Loan_Status'].count()
Self_emp_check=pd.crosstab(columns=training_set['Loan_Status'],values=training_set['Loan_Status'],index=training_set['Self_Employed'],aggfunc='count').apply(lambda x: x/x.sum() ,axis=1)
Self_emp_check
Self_emp_check.plot(kind='bar')
educ_check=pd.crosstab(columns=training_set['Loan_Status'],values=training_set['Loan_Status'],index=training_set['Education'],aggfunc='count').apply(lambda x: x/x.sum() ,axis=1)
educ_check
educ_check.plot(kind='bar')
#Visualizating the data
training_set['TotalIncome'].describe()
sns.boxplot(data=training_set,x='TotalIncome',fliersize=5)
#sns.barplot(data=training_set,x='Loan_Status',y='TotalIncome')
sns.scatterplot(data=training_set,y='LoanAmount',x='TotalIncome',hue='Loan_Status')
plt.ylim(0,1000)
plt.show();
training_set['Loan_Status'].value_counts()
sns.boxplot(data=training_set,x='Loan_Status',y='TotalIncome')
plt.ylim(0,30000)
sns.boxplot(data=training_set,x='Property_Area',y='TotalIncome',hue='Loan_Status') 
plt.ylim(0,40000)
training_set.groupby(['Property_Area','Loan_Status'])['Loan_Status'].count()
Proper_check=pd.crosstab(columns=training_set['Loan_Status'],values=training_set['Loan_Status'],index=training_set['Property_Area'],aggfunc='count').apply(lambda x: x/x.sum(),axis=0)
Proper_check.plot(kind='bar')
training_set['Property_Area']=training_set['Property_Area'].map({'Rural':0,'Semiurban':1,'Urban':2})
plt.figure(figsize=(10,10))
sns.scatterplot(data=training_set,x='LoanAmount',y='TotalIncome',hue='Loan_Status',sizes=5)
plt.figure(figsize=(10,10))
sns.distplot(training_set['TotalIncome'], kde=False)
from sklearn import preprocessing

def label_encode(dataset):
    for cols in dataset.columns:
        if(dataset[cols].dtype == 'object'):
            le=preprocessing.LabelEncoder()
            dataset[cols]=le.fit_transform(dataset[cols])

label_encode(training_set)
label_encode(test_set)

cor=training_set.corr()
cor
plt.figure(figsize=(15,10))
sns.heatmap(cor,annot=np.array(cor),linewidths=0.30,cmap="Blues")
xtrain=training_set.iloc[:,0:11]
ytrain=training_set.iloc[:,11]

from sklearn.linear_model import LogisticRegression
classifier= LogisticRegression()
classifier.fit(xtrain,ytrain)

y_pred=classifier.predict(test_set)
