# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn import preprocessing
import seaborn as sns
#df = pd.read_csv("../input/train_AV3.csv")
df = pd.read_csv("../input/test_AV3.csv")

df.fillna({'Gender':'Male'} ,inplace=True)
df.fillna({'Loan_Amount_Term':360} ,inplace=True)
df.fillna({'Credit_History':1} ,inplace=True)
df.fillna({'LoanAmount':df['LoanAmount'].mean()} ,inplace=True)
df.fillna({'Self_Employed':'No'} ,inplace=True)
df.fillna({'Married':'Yes'} ,inplace=True)
df.fillna({'Dependents':'0'} ,inplace=True)
df['TotIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df.drop(['ApplicantIncome','CoapplicantIncome'], axis=1,inplace = True)

df['Gender'] = df['Gender'].map( {'Male':1, 'Female': 0} ).astype(int)
df['Married'] = df['Married'].map( {'Yes':1, 'No': 0} ).astype(int)
df['Education'] = df['Education'].map( {'Graduate':1, 'Not Graduate': 0} ).astype(int)
df['Self_Employed'] = df['Self_Employed'].map( {'Yes':1, 'No': 0} ).astype(int)
df['Property_Area'] = df['Property_Area'].map( {'Urban':0, 'Rural': 1, 'Semiurban' : 2} ).astype(int)
df['Dependents'] = df['Dependents'].map( {'0':0,'1':1,'2':2, '3+' : 3 } ).astype(int)

#df.drop(df[df['LoanAmount'] == 0].index)
#df.groupby('Dependents').count()
#df.groupby('Credit_History').count()
#df.isnull().sum()
#df.describe()
#df.head()

#g = sns.pairplot(df, hue='Loan_Status')

df_inc_high = df[ df['TotIncome'] > 30000]
df = df.drop(df_inc_high.index, axis=0)

df_lamt_high = df[ df['LoanAmount'] > 350]
df = df.drop(df_lamt_high.index, axis=0)

X = df[['Gender','Married','Dependents','Education','Self_Employed','TotIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area']]
Y = df['Loan_Status']


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20) 

from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
scaler.fit(X_train)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test) 

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=15)
classifier.fit(X_train, y_train) 

y_pred = classifier.predict(X_test)  
from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))  


X = df[['Gender','Married','Dependents','Education','Self_Employed','TotIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area']]

from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
scaler.fit(X)

X = scaler.transform(X)  


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=15)
classifier.fit(X_train, y_train) 

y_pred = classifier.predict(X)  
df['Loan_Status'] = y_pred
df.to_csv('C:/python/Sample_Submission_Av3.csv', sep='\t', encoding='utf-8')

