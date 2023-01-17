import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns


import os
print(os.listdir("../input"))

df = pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv',index_col='customerID')
df.head(15)
df.info()
# We need to convert the Total Charges from object type to Numeric
df['TotalCharges'] = df['TotalCharges'].replace(r'\s+', np.nan, regex=True)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
df.info()
df.Partner.value_counts(normalize=True).plot(kind='bar');
df.SeniorCitizen.value_counts(normalize=True).plot(kind='bar');
df.gender.value_counts(normalize=True).plot(kind='bar');
df.tenure.value_counts(normalize=True).plot(kind='bar',figsize=(16,7));
df.PhoneService.value_counts(normalize=True).plot(kind='bar');
df.MultipleLines.value_counts(normalize=True).plot(kind='bar');
df.InternetService.value_counts(normalize=True).plot(kind='bar');
df.Contract.value_counts(normalize=True).plot(kind='bar');
df.PaymentMethod.value_counts(normalize=True).plot(kind='bar');
# First let's see Our Target Variable
df.Churn.value_counts(normalize=True).plot(kind='bar');

# Now Let's Start Comparing.
# Gender Vs Churn
print(pd.crosstab(df.gender,df.Churn,margins=True))
pd.crosstab(df.gender,df.Churn,margins=True).plot(kind='bar',figsize=(7,5));
print('Percent of Females that Left the Company {0}'.format((939/1869)*100))
print('Percent of Males that Left the Company {0}'.format((930/1869)*100))     
# Contract Vs Churn
print(pd.crosstab(df.Contract,df.Churn,margins=True))
pd.crosstab(df.Contract,df.Churn,margins=True).plot(kind='bar',figsize=(7,5));
print('Percent of Month-to-Month Contract People that Left the Company {0}'.format((1655/1869)*100))
print('Percent of One-Year Contract People that Left the Company {0}'.format((166/1869)*100)) 
print('Percent of Two-Year Contract People that Left the Company {0}'.format((48/1869)*100))     
# Internet Service Vs Churn
print(pd.crosstab(df.InternetService,df.Churn,margins=True))
pd.crosstab(df.InternetService,df.Churn,margins=True).plot(kind='bar',figsize=(7,5));
print('Percent of DSL Internet-Service People that Left the Company {0}'.format((459/1869)*100))
print('Percent of Fiber Optic Internet-Service People that Left the Company {0}'.format((1297/1869)*100)) 
print('Percent of No Internet-Service People that Left the Company {0}'.format((113/1869)*100))     
# Tenure Median Vs Churn
print(pd.crosstab(df.tenure.median(),df.Churn))
pd.crosstab(df.tenure.median(),df.Churn).plot(kind='bar',figsize=(7,5));
# Partner Vs Dependents
print(pd.crosstab(df.Partner,df.Dependents,margins=True))
pd.crosstab(df.Partner,df.Dependents,margins=True).plot(kind='bar',figsize=(5,5));
print('Percent of Partner that had Dependents {0}'.format((1749/2110)*100))
print('Percent of Non-Partner that had Dependents {0}'.format((361/2110)*100))     
# Partner Vs Churn
print(pd.crosstab(df.Partner,df.Churn,margins=True))
pd.crosstab(df.Partner,df.Churn,margins=True).plot(kind='bar',figsize=(5,5));
plt.figure(figsize=(17,8))
sns.countplot(x=df['tenure'],hue=df.Partner);
# Partner Vs Churn
print(pd.crosstab(df.Partner,df.Churn,margins=True))
pd.crosstab(df.Partner,df.Churn,normalize=True).plot(kind='bar');
# Senior Citizen Vs Churn
print(pd.crosstab(df.SeniorCitizen,df.Churn,margins=True))
pd.crosstab(df.SeniorCitizen,df.Churn,normalize=True).plot(kind='bar');
df.boxplot('MonthlyCharges');
df.boxplot('TotalCharges');
df.describe()
# Let's Check the Correaltion Matrix in Seaborn
sns.heatmap(df.corr(),xticklabels=df.corr().columns.values,yticklabels=df.corr().columns.values,annot=True);
# Checking For NULL 
df.isnull().sum()
df.head(15)
fill = df.MonthlyCharges * df.tenure
df.TotalCharges.fillna(fill,inplace=True)
df.isnull().sum()
df.loc[(df.Churn == 'Yes'),'MonthlyCharges'].median()
df.loc[(df.Churn == 'Yes'),'TotalCharges'].median()
df.loc[(df.Churn == 'Yes'),'tenure'].median()
df.loc[(df.Churn == 'Yes'),'PaymentMethod'].value_counts(normalize = True)
df['Is_Electronic_check'] = np.where(df['PaymentMethod'] == 'Electronic check',1,0)
df.loc[(df.Churn == 'Yes'),'PaperlessBilling'].value_counts(normalize = True)
df.loc[(df.Churn == 'Yes'),'DeviceProtection'].value_counts(normalize = True)
df.loc[(df.Churn == 'Yes'),'OnlineBackup'].value_counts(normalize = True)
df.loc[(df.Churn == 'Yes'),'TechSupport'].value_counts(normalize = True)
df.loc[(df.Churn == 'Yes'),'OnlineSecurity'].value_counts(normalize = True)
df= pd.get_dummies(df,columns=['Partner','Dependents',
       'PhoneService', 'MultipleLines','StreamingTV',
       'StreamingMovies','Contract','PaperlessBilling','InternetService'],drop_first=True)
df.info()
df.drop(['StreamingTV_No internet service','StreamingMovies_No internet service'],axis=1,inplace=True)
df.drop('gender',axis=1,inplace=True)
df.drop(['tenure','MonthlyCharges'],axis=1,inplace=True)
df.drop(['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','PaymentMethod'],axis=1,inplace=True)
df = pd.get_dummies(df,columns=['Churn'],drop_first=True)
df.info()
X = df.drop('Churn_Yes',axis=1).as_matrix().astype('float')
y = df['Churn_Yes'].ravel()
# train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)
# Import Logistic Regression
from sklearn.linear_model import LogisticRegression
# create model
model_lr_1 = LogisticRegression(random_state=0)
# train model
model_lr_1.fit(X_train,y_train)
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score,classification_report
# performance metrics
# accuracy
print ('accuracy for logistic regression - version 1 : {0:.2f}'.format(accuracy_score(y_test, model_lr_1.predict(X_test))))
# confusion matrix
print ('confusion matrix for logistic regression - version 1: \n {0}'.format(confusion_matrix(y_test, model_lr_1.predict(X_test))))
# precision 
print ('precision for logistic regression - version 1 : {0:.2f}'.format(precision_score(y_test, model_lr_1.predict(X_test))))
# precision 
print ('recall for logistic regression - version 1 : {0:.2f}'.format(recall_score(y_test, model_lr_1.predict(X_test))))
print(classification_report(y_test,model_lr_1.predict(X_test)))
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)
# performance metrics
# accuracy
print ('accuracy for xgboost- version 1 : {0:.2f}'.format(accuracy_score(y_test, classifier.predict(X_test))))
# confusion matrix
print ('confusion matrix for xgboost - version 1: \n {0}'.format(confusion_matrix(y_test, classifier.predict(X_test))))
# precision 
print ('precision for xgboost - version 1 : {0:.2f}'.format(precision_score(y_test, classifier.predict(X_test))))
# precision 
print ('recall for xgboost - version 1 : {0:.2f}'.format(recall_score(y_test, classifier.predict(X_test))))
print(classification_report(y_test,classifier.predict(X_test)))
