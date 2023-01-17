# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')
data.head()
data.describe()
print('Unique values for below columns from dataset:')

print('========='*10)

print(data['Attrition'].unique())

print(data['Over18'].unique())

print(data['OverTime'].unique())

print('========='*10)
data['Attrition'] = data['Attrition'].apply(lambda x:1 if x == 'Yes' else 0)

data['Over18'] = data['Over18'].apply(lambda x:1)

data['OverTime'] = data['OverTime'].apply(lambda x:1 if x == 'Yes' else 0)
data.head()
sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
data.hist(bins=30,figsize=(20,20),color='b')

plt.show()
data.columns
data.drop(['EmployeeNumber','Over18','StandardHours','EmployeeCount'],axis=1,inplace=True)
fig = plt.figure(figsize=(20,2))

sns.countplot(y = data['Attrition'])

plt.title('Count for employee attrition')

plt.show()
df_left = data[data['Attrition'] == 1]

df_stayes = data[data['Attrition'] == 0]
print(f'Percentage of People left the Company : {(len(df_left) / len(data))*100} %')

print(f'Percentage of People Stayed the Company : {(len(df_stayes) / len(data))*100} %')
#let's check the correlation between the data

plt.figure(figsize=(20,20))

sns.heatmap(data.corr(),cmap='viridis',annot=True,fmt='.2f')

plt.show()
plt.figure(figsize=(20,10))

sns.countplot(data['Age'],hue=data['Attrition'])

plt.show()
plt.figure(figsize=[20,20])



plt.subplot(411)

sns.countplot(x=data['JobSatisfaction'],hue=data['Attrition'])

plt.subplot(412)

sns.countplot(x=data['MaritalStatus'],hue=data['Attrition'])

plt.subplot(413)

sns.countplot(x=data['JobRole'],hue=data['Attrition'])

plt.subplot(414)

sns.countplot(x=data['JobLevel'],hue=data["Attrition"])

# plt.subplot(415)

# sns.countplot(x=data['JobInvolvement'],hue=data["Attrition"])

plt.show()
plt.figure(figsize=(12,7))



sns.kdeplot(df_left['DistanceFromHome'],label='Employees who left',shade='True',color='r')

sns.kdeplot(df_stayes['DistanceFromHome'],label='Employees who Stayes',shade='True',color='b')

plt.xlabel('Distance from Home')

plt.show()
plt.figure(figsize=(12,7))



sns.kdeplot(df_left['TotalWorkingYears'],label='Employees who left',shade='True',color='r')

sns.kdeplot(df_stayes['TotalWorkingYears'],label='Employees who Stayes',shade='True',color='b')

plt.xlabel('Total working years')

plt.show()
plt.figure(figsize=(12,7))



sns.kdeplot(df_left['YearsWithCurrManager'],label='Employees who left',shade='True',color='r')

sns.kdeplot(df_stayes['YearsWithCurrManager'],label='Employees who Stayes',shade='True',color='b')

plt.xlabel('Years with current manager')

plt.show()
sns.boxplot(x= 'MonthlyIncome',y='Gender',data=data)
sns.boxplot(x='MonthlyIncome',y='JobRole',data=data)
x_cat = data.select_dtypes(include='object')

x_cat
from sklearn.preprocessing import OneHotEncoder

onehotencoder = OneHotEncoder()

x_cat = onehotencoder.fit_transform(x_cat).toarray()
x_cat = pd.DataFrame(x_cat)

x_cat
x_numerical = data.select_dtypes(exclude='object')

x_numerical.drop('Attrition',axis=1,inplace=True)

x_numerical
x_data = pd.concat([x_cat,x_numerical],axis=1)

x_data
from sklearn.preprocessing import MinMaxScaler

scalar = MinMaxScaler()

x = scalar.fit_transform(x_data)
x
y = data['Attrition']

y
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.25)
print(f'Shape of X train :{X_train.shape}')

print(f'Shape of X test :{X_test.shape}')

print(f'Shape of y train :{y_train.shape}')

print(f'Shape of y test :{y_test.shape}')
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
model = LogisticRegression()

model.fit(X_train,y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report

print(f'Accuracy : {100 * accuracy_score(y_pred,y_test)}')
cm = confusion_matrix(y_pred,y_test)

sns.heatmap(cm,annot=True,fmt='d')
print('Classification report')

print('======='*10)

print(classification_report(y_pred,y_test))

print('======='*10)
from sklearn.ensemble import RandomForestClassifier



randomforest = RandomForestClassifier()

randomforest.fit(X_train,y_train)
y_pred = randomforest.predict(X_test)
cm = confusion_matrix(y_pred,y_test)

sns.heatmap(cm,annot=True,fmt='d')
print('Classification report')

print('======='*10)

print(classification_report(y_pred,y_test))

print('======='*10)
from xgboost import XGBClassifier



xgb = XGBClassifier()

xgb.fit(X_train,y_train)
y_pred = xgb.predict(X_test)
cm = confusion_matrix(y_pred,y_test)

sns.heatmap(cm,annot=True,fmt='d')
print('Classification report')

print('======='*10)

print(classification_report(y_pred,y_test))

print('======='*10)
from imblearn.over_sampling import SMOTE



sm = SMOTE(random_state=27,sampling_strategy=1.0)

X_train,y_train = sm.fit_sample(X_train,y_train)
smote_logistic = LogisticRegression()

smote_logistic.fit(X_train,y_train)
smote_pred = smote_logistic.predict(X_test)
sns.heatmap(confusion_matrix(y_test,smote_pred),annot=True,fmt='.2f',cmap='YlGnBu')

plt.savefig('rand_after_oversample.png')

plt.show()
print('Classification report')

print('======='*10)

print(classification_report(smote_pred,y_test))

print('======='*10)
randomforest = RandomForestClassifier()

randomforest.fit(X_train,y_train)
y_pred_smote = randomforest.predict(X_test)
sns.heatmap(confusion_matrix(y_pred_smote,y_test),annot=True,fmt='d',cmap='YlGnBu')

plt.savefig('rand_after_oversample.png')

plt.show()