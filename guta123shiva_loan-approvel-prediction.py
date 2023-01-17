import pandas as pd
import numpy as np
import statistics as stat
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report,roc_auc_score,roc_curve,auc
import statsmodels.api as sm
df=pd.read_csv('test_Y.csv')
df1=pd.read_csv('train_u.csv')
df.head(5)
df1.head(5)
print(df.columns)
print(df1.columns)
#summery of numerical variable for training dataset
df1.describe()
#Types of variable
df1.dtypes
#Getting the percentage value for the Property_area
r=pd.get_dummies(df1['Property_Area'])
r1=(sum(r["Semiurban"]),sum(r["Urban"]),sum(r["Rural"]))
plt.pie(r1,labels=["Semiurban","Urban","Rural"],shadow=True,explode=(.1,.1,.1), autopct='%1.1f%%')
#Get, how many male and feamale take the loan
r2=pd.get_dummies(df1['Gender'])
r3=(sum(r2["Female"]),sum(r2["Male"]))
plt.pie(r3,labels=["Male","Female"],shadow=True,explode=(.1,.1), autopct='%1.1f%%')
#Get,percent value for variable Education
p=pd.get_dummies(df1['Education'])
p1=(sum(p["Graduate"]),sum(p["Not Graduate"]))
plt.pie(p1,labels=["Graduate","Not Graduate"],shadow=True,explode=(.1,.1), autopct='%1.1f%%')
sns.distplot(df1['ApplicantIncome'],bins=100)
df1['ApplicantIncome'].plot(kind="kde",figsize=(20,5))
sns.catplot(data=df1,x='Education',y='ApplicantIncome',hue='Loan_Status',kind='box')
#df1.boxplot(column='ApplicantIncome',by='Education')
sns.catplot(data=df1,x='Education',y='LoanAmount',hue='Loan_Status',kind='boxen')
sns.catplot(data=df1,x='Married',y='ApplicantIncome',hue='Loan_Status',kind='box')
df1['LoanAmount_log']=np.log(df1['LoanAmount'])
sns.distplot(df1['LoanAmount_log'],bins=20)
sns.boxplot(data=df1,y='LoanAmount',x='Education',hue='Loan_Status')
sns.boxplot(data=df1,y='LoanAmount',x='Married',hue='Loan_Status')
sns.boxplot(data=df1,y='LoanAmount',x='Gender',hue='Loan_Status')
sns.distplot(df['LoanAmount'],bins=20)
loan_app=df1['Loan_Status'].value_counts()["Y"]
print(loan_app)
print(pd.crosstab(df1["Credit_History"],df1["Loan_Status"],margins=True))
plt.figure(figsize=(10,4))
sns.countplot(x='Credit_History',hue='Loan_Status',data=df1,order=df1['Credit_History'].value_counts().index);
plt.figure(figsize=(10,4))
sns.countplot(x='Education',hue='Loan_Status',data=df1,order=df1['Education'].value_counts().index);
plt.figure(figsize=(10,4))
sns.countplot(x='Gender',hue='Loan_Status',data=df1,order=df1['Gender'].value_counts().index);
def percentagecon(ser):
    return ser/float(ser[-1])
df_1=pd.crosstab(df1["Credit_History"],df1["Loan_Status"],margins=True).apply(percentagecon,axis=1)
loan_app_wcredit_1=df_1["Y"][1]
print(loan_app_wcredit_1*100)
plt.plot(df1['ApplicantIncome'])
plt.plot(df1['CoapplicantIncome'])
df1['TotalIncome']=df1['ApplicantIncome']+df1['CoapplicantIncome']
sns.distplot(df1['TotalIncome'],bins=20)
cat=['Gender','Married','Dependents','Education','Self_Employed','Credit_History','Property_Area','Loan_Status']
for var in cat:
    le=preprocessing.LabelEncoder()
    df1[var]=le.fit_transform(df1[var].astype('str'))
df1.dtypes
x2=['LoanAmount','ApplicantIncome','CoapplicantIncome','Loan_Amount_Term']
for i in x2:
    df1[i].fillna(df1[i].mean(),inplace=True)
df1['Credit_History'].fillna(df1['Credit_History'].mode()[0],inplace=True)
#we choose dependent and independent variables
x=df1[['Credit_History','Education','Gender']]
y=df1['Loan_Status']
#Summery of logistic Regression model
import statsmodels.api as sm
x1=sm.add_constant(x)
logist_modal=sm.Logit(y,x1)
result=logist_modal.fit()
print(result.summary())
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.20)
log_reg=LogisticRegression()
log_reg.fit(x_train,y_train)
y_pred=log_reg.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)
tp,fp,fn,tn=confusion_matrix(y_test,y_pred).ravel()
print("True_positive",tp)
print("False_positive",fp)
print("False_negative",fn)
print("True_negative",tn)
print(classification_report(y_test,y_pred))
y_prob_train=log_reg.predict_proba(x_train)[:,1]
y_prob_train.reshape(1,-1)
fpr_p,tpr_p,threshol=roc_curve(y_train,y_prob_train)
roc_auc_p=auc(fpr_p,tpr_p)
print(roc_auc_p)
plt.figure()
plt.plot(fpr_p,tpr_p,color='green',label='ROC curve'% roc_auc_p)
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Recurcive Operating characterstics')
plt.legend(loc='lower right')
plt.show()
df1['Type']='Train'
df['Type']='Test'
N_data=pd.concat([df1,df],axis=0,sort=True)
N_data.isnull().sum()
idcol=['Loan_ID']
tar_col=["Loan_Status"]
new_col=['Credit_History','Dependents','Gender','Married','Education','Property_Area','Self_Employed']
for var in new_col:
    num=LabelEncoder()
    N_data[var]=num.fit_transform(N_data[var].astype('str'))
    
train_mod=N_data[N_data['Type']=='Train']
test_mod=N_data[N_data['Type']=='Test']
train_mod["Loan_Status"]=num.fit_transform(train_mod["Loan_Status"])
N_data['TotalIncome']=N_data['ApplicantIncome']+N_data['CoapplicantIncome']
N_data['TotalIncome_log']=np.log(N_data['TotalIncome'])
sns.distplot(N_data['TotalIncome_log'],bins=20)
predict_log=['Credit_History','Education','Gender']
x_train_c=train_mod[predict_log]
y_train_c=train_mod['Loan_Status']
x_test_c=test_mod[predict_log]
result_c=log_reg.fit(x_train_c,y_train_c)
predicted_c=log_reg.predict(x_test_c)
predicted_c=num.inverse_transform(predicted_c)
test_mod['Loan_Status']=predicted_c
y_c=y[:367]
accuracy_c=accuracy_score(predicted_c,y_c)
print(accuracy_c)
