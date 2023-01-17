import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
data= pd.read_csv("/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")
data
data.info()
!pip install pycaret
from pycaret.classification import *
print ("Rows     : " ,data.shape[0])
print ("Columns  : " ,data.shape[1])
print ("\nFeatures : \n" ,data.columns.tolist())
print ("\nMissing values :  ", data.isnull().sum().values.sum())
print ("\nUnique values :  \n",data.nunique())
data['TotalCharges']=data["TotalCharges"].replace(r'\s+',np.nan,regex=True)
data['TotalCharges']=pd.to_numeric(data['TotalCharges'])
data.Partner.value_counts(normalize=True).plot(kind='bar')
data.SeniorCitizen.value_counts(normalize=True).plot(kind='bar')
data.gender.value_counts(normalize=True).plot(kind='bar')
data.tenure.value_counts(normalize=True).plot(kind='bar',figsize=(16,7))
data.PhoneService.value_counts(normalize=True).plot(kind='bar')
data.MultipleLines.value_counts(normalize=True).plot(kind='bar')
data.InternetService.value_counts(normalize=True).plot(kind='bar')
data.Contract.value_counts(normalize=True).plot(kind='bar')
data.PaymentMethod.value_counts(normalize=True).plot(kind='bar')
data.Churn.value_counts(normalize=True).plot(kind='bar')
print(pd.crosstab(data.gender,data.Churn,margins=True))
pd.crosstab(data.gender,data.Churn,margins=True).plot(kind='bar',figsize=(7,5))
print('Percent of females that left the company {0}'.format((939/1869)*100))
print('Percent of males that left the company {0}'.format((930/1869)*100))
print(pd.crosstab(data.Contract,data.Churn,margins=True))
pd.crosstab(data.Contract,data.Churn,margins=True).plot(kind='bar',figsize=(7,5))
print("% off month to month ",((1655/1869)*100))
print("% off one year ",((166/1869)*100))
print("% off two year ",((48/1869)*100))
print(pd.crosstab(data.InternetService,data.Churn,margins=True))
pd.crosstab(data.InternetService,data.Churn,margins=True).plot(kind='bar',figsize=(7,5))
print("% of DSL service",((459/1869)*100))
print("% of fibre optic",((1297/1869)*100))
print("% of No internet",((113/1869)*100))
print(pd.crosstab(data.tenure.median(),data.Churn,margins=True))
pd.crosstab(data.tenure.median(),data.Churn,margins=True).plot(kind='bar',figsize=(7,5))
print(pd.crosstab(data.Partner,data.Dependents,margins=True))
pd.crosstab(data.Partner,data.Dependents,margins=True).plot(kind='bar',figsize=(7,5))
print("% of partner that had dependents",((1749/2110)*100))
print("% of non-partner that had dependents",((361/2110)*100))
print(pd.crosstab(data.Partner,data.Churn,margins=True))
pd.crosstab(data.Partner,data.Churn,margins=True).plot(kind='bar',figsize=(7,5))
plt.figure(figsize=(17,8))
sns.countplot(x=data['tenure'],hue=data.Partner)
print(pd.crosstab(data.SeniorCitizen,data.Churn,margins=True))
pd.crosstab(data.SeniorCitizen,data.Churn,margins=True).plot(kind='bar',figsize=(7,5))
data.boxplot('MonthlyCharges')
data.boxplot('TotalCharges')
data.describe()
data.isnull().sum()
fill=data.MonthlyCharges*data.tenure
data.TotalCharges.fillna(fill,inplace=True)
data.isnull().sum()
data.loc[(data.Churn=='Yes'),'MonthlyCharges'].median()
data.loc[(data.Churn=='Yes'),'TotalCharges'].median()
data.loc[(data.Churn=='Yes'),'tenure'].median()
data.loc[(data.Churn=='Yes'),'PaymentMethod'].value_counts(normalize=True)
df=data
def changeColumnsToString(df):
    columnsNames=['Partner','Dependents','PhoneService','MultipleLines','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling','Churn']
    for col in columnsNames:
        df[col]=df[col].astype('str').str.replace('Yes','1').replace('No','0').replace('No internet service','0').replace('No phone service',0)

changeColumnsToString(df)

df['SeniorCitizen']=df['SeniorCitizen'].astype(bool)
df['TotalCharges']=pd.to_numeric(df['TotalCharges'],errors='coerce')
df.head(2)

print("Payment methods: ",df.PaymentMethod.unique())
print("Contract types: ",df.Contract.unique())
print("Gender: ",df.gender.unique())
print("Senior Citizen: ",df.SeniorCitizen.unique())
print("Internet Service Types: ",df.InternetService.unique())

df['gender']=df['gender'].astype('category')
df['PaymentMethod']=df['PaymentMethod'].astype('category')
df['Contract']=df['Contract'].astype('category')
df['SeniorCitizen']=df['SeniorCitizen'].astype('category')
df['InternetService']=df['InternetService'].astype('category')
df.dtypes
dfPaymentDummies = pd.get_dummies(df['PaymentMethod'], prefix = 'payment')
dfContractDummies = pd.get_dummies(df['Contract'], prefix = 'contract')
dfGenderDummies = pd.get_dummies(df['gender'], prefix = 'gender')
dfSeniorCitizenDummies = pd.get_dummies(df['SeniorCitizen'], prefix = 'SC')
dfInternetServiceDummies = pd.get_dummies(df['InternetService'], prefix = 'IS')

print(dfPaymentDummies.head(3))
print(dfContractDummies.head(3))
print(dfGenderDummies.head(3))
print(dfSeniorCitizenDummies.head(3))
print(dfInternetServiceDummies.head(3))

df.drop(['gender','PaymentMethod','Contract','SeniorCitizen','InternetService'], axis=1, inplace=True)

df = pd.concat([df, dfPaymentDummies], axis=1)
df = pd.concat([df, dfContractDummies], axis=1)
df = pd.concat([df, dfGenderDummies], axis=1)
df = pd.concat([df, dfSeniorCitizenDummies], axis=1)
df = pd.concat([df, dfInternetServiceDummies], axis=1)
df.head(2)

df.columns = ['customerID', 'Partner', 'Dependents', 'tenure', 'PhoneService',
       'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling',
       'MonthlyCharges', 'TotalCharges', 'Churn',
       'payment_Bank_transfer_auto', 'payment_Credit_card_auto',
       'payment_Electronic_check', 'payment_Mailed_check',
       'contract_Month_to_month', 'contract_One_year', 'contract_Two_year',
       'gender_Female', 'gender_Male', 'SC_False', 'SC_True', 'IS_DSL',
       'IS_Fiber_optic', 'IS_No']

numericColumns=np.array(['Partner', 'Dependents', 'tenure', 'PhoneService',
       'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling',
       'MonthlyCharges', 'TotalCharges', 'Churn',
       'payment_Bank_transfer_auto', 'payment_Credit_card_auto',
       'payment_Electronic_check', 'payment_Mailed_check',
       'contract_Month_to_month', 'contract_One_year', 'contract_Two_year',
       'gender_Female', 'gender_Male', 'SC_False', 'SC_True', 'IS_DSL',
       'IS_Fiber_optic', 'IS_No'])

for columnName in numericColumns:
    df[columnName]=pd.to_numeric(df[columnName],errors='coerce')
df.dtypes
df
train=df[:6000]
train
test=df[6000:]
test
new_test=test['Churn']
test.drop(['Churn'],axis=1,inplace=True)
test
from pycaret.classification import *
clf = setup(data = train, 
             target = 'Churn'
           )
compare_models()
lgbm  = create_model('lightgbm')    
tuned_lightgbm = tune_model('lightgbm')

plot_model(estimator = tuned_lightgbm, plot = 'learning')
plot_model(estimator = tuned_lightgbm, plot = 'auc')
plot_model(estimator = tuned_lightgbm, plot = 'confusion_matrix')
plot_model(estimator = tuned_lightgbm, plot = 'feature')
evaluate_model(tuned_lightgbm)
interpret_model(tuned_lightgbm)
predict_model(tuned_lightgbm, data=test)
predictions = predict_model(tuned_lightgbm, data=test)
predictions.head(20)
new_test3=round(predictions['Score']).astype(int)
new_test3
new_test

new_test.to_csv('submission1.csv',index=False)

new_test3.to_csv('submission2.csv',index=2)
d=pd.read_csv('submission1.csv')
d1=pd.read_csv('submission2.csv')
d1
d['pred_churn']=d1['0.1']
d.to_csv('final_sub.csv')
d
