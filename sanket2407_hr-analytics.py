import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import StandardScaler

import warnings

warnings.filterwarnings('ignore')
df=pd.read_csv('../input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')
df.head()
df.shape
df.tail(10)
df.isnull().sum()
df.info()
df.describe()
df.fillna(0,inplace=True)
df.isnull().sum()
df.drop(['EmployeeCount','EmployeeNumber','StandardHours'],inplace=True,axis=1)
df.head()
df.corr()
plt.figure(figsize=[20,8])

sns.heatmap(df.corr(),cmap='BuGn',annot=True)

plt.show()
df['Attrition'].value_counts()

print('Yes Attrition percentage is:',(len(df[df['Attrition']=='Yes'])/len(df))*100)

print('Yes Attrition percentage is:',(len(df[df['Attrition']=='No'])/len(df))*100)
df.columns
sns.countplot(df['Attrition'])

plt.show()
sns.countplot(df['BusinessTravel'])

plt.show()
sns.countplot(df['Department'])

plt.show()
sns.distplot(df['DistanceFromHome'])

plt.show()
df['DistanceFromHome'].mean()
sns.countplot(df['Education'])

plt.show()
plt.figure(figsize=[15,6])

sns.countplot(df['EducationField'])

plt.show()
sns.countplot(df['Gender'])

plt.show()
sns.countplot(df['JobLevel'])

plt.show()
plt.figure(figsize=[19,5])

sns.countplot(df['JobRole'])

plt.show()
sns.countplot(df['MaritalStatus'])

plt.show()
df.groupby(['Attrition'])['MonthlyIncome'].mean()
sns.countplot(df['NumCompaniesWorked'])

plt.show()
sns.countplot(data=df,x='Gender',hue='Attrition')

plt.show
sns.countplot(df['StockOptionLevel'])

plt.show()
sns.countplot(data=df,x='Attrition',hue='StockOptionLevel')

plt.show()
sns.distplot(df['TotalWorkingYears'])

plt.show()
sns.countplot(data=df,x='Attrition',hue='JobRole')

plt.show()
sns.countplot(data=df,x='Attrition',hue='JobLevel')

plt.show()
df.info()
from sklearn.preprocessing import LabelEncoder

labelEncoder_X = LabelEncoder()

df['BusinessTravel'] = labelEncoder_X.fit_transform(df['BusinessTravel'])

df['Department'] = labelEncoder_X.fit_transform(df['Department'])

df['EducationField'] = labelEncoder_X.fit_transform(df['EducationField'])

df['Gender'] = labelEncoder_X.fit_transform(df['Gender'])

df['JobRole'] = labelEncoder_X.fit_transform(df['JobRole'])

df['MaritalStatus'] = labelEncoder_X.fit_transform(df['MaritalStatus'])

df['Over18'] = labelEncoder_X.fit_transform(df['Over18'])

df['OverTime']=labelEncoder_X.fit_transform(df['OverTime'])
labelencoder_y=LabelEncoder()

df['Attrition']=labelencoder_y.fit_transform(df['Attrition'])
df.head()
y=df['Attrition']

X=df.drop('Attrition',axis=1)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)
X_train.head()
X_train.columns
X_train.info()
X_test.head()
from sklearn.preprocessing import StandardScaler

Scaler_X = StandardScaler()

X_train = Scaler_X.fit_transform(X_train)

X_test = Scaler_X.transform(X_test)
#import some comman libs:

from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix, accuracy_score
lg=LogisticRegression()

lg.fit(X_train,y_train)
pred=lg.predict(X_test)
print(accuracy_score(y_test,pred))

print(confusion_matrix(y_test,pred))
from sklearn.metrics import classification_report

print(classification_report(y_test,pred))