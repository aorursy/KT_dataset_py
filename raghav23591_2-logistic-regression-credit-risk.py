import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import pearsonr,spearmanr

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,f1_score,recall_score,classification_report

%matplotlib inline
df=pd.read_csv("../input/train.csv")
df.head(5)
df.dtypes
df.isna().sum()
d=df.dropna()
print("before",df.shape[0])

print("after",d.shape[0])

df.shape[0]-d.shape[0]
d.columns
d.Dependents.unique()
sns.distplot(d.ApplicantIncome)
sns.distplot(d.CoapplicantIncome)
sns.distplot(d.LoanAmount)
sns.boxplot(y=d.LoanAmount)
d.LoanAmount.mean()
d.LoanAmount.describe()
d.Loan_Amount_Term.unique()
d.Loan_Amount_Term.value_counts()
d.Credit_History.unique()

#d['Credit_History'].unique()
d.Property_Area.unique()

#d['Property_Area'].unique()
d.Loan_Status.unique()
d.Loan_Status.value_counts()
# H0:Applicant income is very important. So that may impact the loan status

d.groupby('Loan_Status')['ApplicantIncome'].mean()
sns.scatterplot(d.ApplicantIncome,d.LoanAmount)
print('spearmanr:',spearmanr(d.ApplicantIncome,d.LoanAmount)[0])

print('pearsonr:',pearsonr(d.ApplicantIncome,d.LoanAmount)[0])
sns.scatterplot(d.ApplicantIncome,d.CoapplicantIncome)
#Creating a new column called 'Total_Income' by combining (Adding) Applicant_Income and CoApplicant_Income

d['Total_Income']=d.ApplicantIncome+d.CoapplicantIncome
print('spearmanr:',spearmanr(d.Total_Income,d.LoanAmount)[0])

print('pearsonr:',pearsonr(d.Total_Income,d.LoanAmount)[0])
sns.lmplot(x='Total_Income',y='LoanAmount',data=d,hue='Loan_Status',fit_reg=False)
print('Pearsonr:',pearsonr(d[d.Total_Income<30000].Total_Income,d[d.Total_Income<30000].LoanAmount)[0])

print('Spearmanr:',spearmanr(d[d.Total_Income<30000].Total_Income,d[d.Total_Income<30000].LoanAmount)[0])



sns.lmplot(x='Total_Income',y='LoanAmount',data=d[d.Total_Income<30000],hue='Loan_Status',fit_reg=False)
# H0: Self Employed is related to Loan Status

d.groupby('Self_Employed')['Loan_Status'].value_counts()
d.groupby('Self_Employed')['Loan_Status'].count()
d.groupby('Self_Employed')['Loan_Status'].value_counts()/d.groupby('Self_Employed')['Loan_Status'].count()

# To know the probability, we can divide by counts()
#H0: Education and Loan Status are related to each other.. 

d.groupby('Education')['Loan_Status'].value_counts()
d.groupby('Education')['Loan_Status'].count()
d.groupby('Education')['Loan_Status'].value_counts()/d.groupby('Education')['Loan_Status'].count()
#H0: Education and Self Employed are related to each other

d.groupby('Education')['Self_Employed'].value_counts()/d.groupby('Education')['Loan_Status'].count()
#H0: Gender and Loan status are related to each other. 

d.groupby('Gender')['Loan_Status'].value_counts()/d.groupby('Gender')['Loan_Status'].count()
#H0: Married and Loan status are related to each other. 

d.groupby('Married')['Loan_Status'].value_counts()/d.groupby('Married')['Loan_Status'].count()

#H0: Dependants and Loan Status are related to each other. 

d.groupby('Dependents')['Loan_Status'].value_counts()/d.groupby('Dependents')['Loan_Status'].count()
# To find the mean of the Total Income who has Dependants

d.groupby('Dependents')['Total_Income'].mean()
# To find the mean of the Applicant Income who has Dependants



d.groupby('Dependents')['ApplicantIncome'].mean()
#H0: Property Area and Loan Status are related to each other. 

d.groupby('Property_Area')['Loan_Status'].value_counts()/d.groupby('Property_Area')['Loan_Status'].count()
d.columns
#Making the Data set ordinal:::



def datacleaning(x):

    x.Gender=x.Gender.map(lambda x:1 if x=='Male' else 0) #Assigning 1 if Gender is 'Male' and 0 if Gemder is 'Female'

    x.Married=x.Married.map(lambda x:1 if x=='Yes' else 0) #Assigning 1 if Married is 'Yes' and 0 if Married is 'No'

    x.Dependents=x.Dependents.map(lambda x:3 if x=='3+' else int(x)) #Assigning 3 if Dependents is '3+'' and same values if other than '3+'

    x.Education=x.Education.map(lambda x:1 if x=='Graduate' else 0) #Assigning 1 if Education is 'Graduate' and o if education is 'Not graduate'

    x.Self_Employed=x.Self_Employed.map(lambda x:1 if x=='Yes' else 0) #Assigning 1 if SelfEmployed is 'Yes' and 0 if selfEmployed is 'No'

    dummies=pd.get_dummies(x.Property_Area) #Get Dummies will create columns(=Unique values) and assign 1 and 0. 

    x=x.join(dummies) #Joining dummies to the dataset

    x["TotalIncome"]=x.ApplicantIncome+x.CoapplicantIncome

    y=x.Loan_Status.map(lambda x:1 if x=='Y' else 0) #Assigning 1 if the Loan status is 'Y' and 0 if the Loan Status is 'N'

    x=x.drop(['Loan_ID','LoanAmount','Loan_Amount_Term','Property_Area','Loan_Status','ApplicantIncome','CoapplicantIncome'],axis=1) #Dropping the unwanted columns from the dataset

    return x,y

   
X,y=datacleaning(d.copy())
X.head(10)
X.Gender.value_counts()
lr=LogisticRegression()

lr.fit(X,y)
#Accuracy Score

lr.score(X,y)
y_hat=lr.predict(X)
c=0

for i,j in zip(y,y_hat):

    if i==j:

        c+=1

print(c/len(y))
accuracy_score(y,y_hat)
y_hat
for i,j in zip(y,y_hat):

    print(i,j)
print(y.value_counts())

confusion_matrix(y,y_hat,labels=[1,0])
# Accuracy works best if false positives and false negatives have similar cost. If the cost of false positives and false negatives are very different, it’s better to look at both Precision and Recall.



print("precision:",precision_score(y,y_hat)) #Precision = TP/TP+FP.. High precision relates to the low false positive rate

print("recall:",recall_score(y,y_hat)) #Recall(Sensitivity) = TP/TP+FN.. Of all predicted, how many did we label. We have got recall of 0.978 which is good for this model as it’s above 0.5.

print("f1_score:",f1_score(y,y_hat)) #F1 Score = 2*(Recall * Precision) / (Recall + Precision)... F1 is usually more useful than accuracy, especially if you have an uneven class distribution..
print(classification_report(y,y_hat))
lr.predict(X)
lr.predict_proba(X)
pr_pass=lr.predict_proba(X)[:,1]
y_hat_thr=[]

for i in pr_pass:

    if i>0.72:

        y_hat_thr.append(1)

    else:

        y_hat_thr.append(0)

        
print('Confusion Matrix after setting the Threshold:\n',confusion_matrix(y,y_hat_thr,labels=[1,0]))

print('\n\nClassification report after setting the Threshold:\n',classification_report(y,y_hat_thr))
for i,j in zip(lr.coef_[0],X.columns):

    print(i,'*',j,'+')

print("\nIntercept",lr.intercept_[0])
d.index
df.index
test_index=df.index.difference(d.index)
test_index
test=df.loc[test_index]
test.head(5)
test.isna().sum()
d['Credit_History'].value_counts()
test.Gender=test.Gender.fillna("Male")

test.Married=test.Married.fillna("Yes")

test.Dependents=test.Dependents.fillna("0")

test.Self_Employed=test.Self_Employed.fillna("No")

test.LoanAmount=test.LoanAmount.fillna(d.LoanAmount.median())

test.Loan_Amount_Term=test.Loan_Amount_Term.fillna(d.Loan_Amount_Term.median())

test.Credit_History=test.Credit_History.fillna(1)
test.isna().sum()
X_test,y_test=datacleaning(test)
X_test.columns
lr.score(X_test,y_test)
y_hat_test=lr.predict(X_test)

print(confusion_matrix(y_test,y_hat_test,labels=[1,0]))
print(confusion_matrix(y,y_hat,labels=[1,0]))
#Lets Compare the Precision,Recall,F1-Score of both Training and Testing Data.



print("Train Data:\n",classification_report(y,y_hat))

print("Test Data:\n",classification_report(y_test,y_hat_test))
y_hat_test_th=[]

pr_pass_test=lr.predict_proba(X_test)[:,1]

for i  in pr_pass_test:

    if i>0.72:

        y_hat_test_th.append(1)

    else:

        y_hat_test_th.append(0)
#Lets Compare the Precision,Recall,F1-Score of both Training and Testing Data after setting the Threshold value.



print("Train Data after setting the threshold:\n",classification_report(y,y_hat_thr))

print("\nTest Data after setting the threshold:\n",classification_report(y_test,y_hat_test_th))