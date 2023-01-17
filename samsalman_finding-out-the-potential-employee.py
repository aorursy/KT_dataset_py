import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data=pd.read_csv('/kaggle/input/hr-analytics-case-study/general_data.csv')
df=data.copy()
df.head()
df.isna().sum()
df.fillna(0,inplace=True)
df.drop(['EmployeeCount','EmployeeID','StandardHours'],axis=1, inplace=True)
import matplotlib.pyplot as plt

import seaborn as sns
df.columns
corr_cols=df[['Age', 'Attrition', 'BusinessTravel', 'Department', 'DistanceFromHome','Education',\

              'EducationField', 'Gender', 'JobLevel', 'JobRole','MaritalStatus',\

              'MonthlyIncome', 'NumCompaniesWorked', 'Over18','PercentSalaryHike',\

              'StockOptionLevel', 'TotalWorkingYears','TrainingTimesLastYear',\

              'YearsAtCompany', 'YearsSinceLastPromotion','YearsWithCurrManager']]
corr=corr_cols.corr()

plt.figure(figsize=(16,7))

sns.heatmap(corr,annot=True)

plt.show()
sns.countplot(x='Attrition',data=df)
sns.countplot(x='Attrition',data=df,hue='Gender')
sns.countplot(x='Attrition',data=df,hue='JobLevel')
def Age(dataframe):

    dataframe.loc[dataframe['Age'] <= 30,'Age'] = 1

    dataframe.loc[(dataframe['Age'] > 30) & (dataframe['Age'] <= 40), 'Age'] = 2

    dataframe.loc[(dataframe['Age'] > 40) & (dataframe['Age'] <= 50), 'Age'] = 3

    dataframe.loc[(dataframe['Age'] > 50) & (dataframe['Age'] <= 60), 'Age'] = 4

    return dataframe



Age(df); 
sns.countplot(x='Attrition',data=df,hue='Age')
print(df['BusinessTravel'].unique())

print(df['Department'].unique())

print(df['EducationField'].unique())

print(df['Gender'].unique())

print(df['JobRole'].unique())

print(df['MaritalStatus'].unique())

print(df['Over18'].unique())
from sklearn.preprocessing import LabelEncoder

labelEncoder_X = LabelEncoder()

df['BusinessTravel'] = labelEncoder_X.fit_transform(df['BusinessTravel'])

df['Department'] = labelEncoder_X.fit_transform(df['Department'])

df['EducationField'] = labelEncoder_X.fit_transform(df['EducationField'])

df['Gender'] = labelEncoder_X.fit_transform(df['Gender'])

df['JobRole'] = labelEncoder_X.fit_transform(df['JobRole'])

df['MaritalStatus'] = labelEncoder_X.fit_transform(df['MaritalStatus'])

df['Over18'] = labelEncoder_X.fit_transform(df['Over18'])
from sklearn.preprocessing import LabelEncoder

label_encoder_y=LabelEncoder()

df['Attrition']=label_encoder_y.fit_transform(df['Attrition'])
df.head()
corr_cols = df[['Age','Attrition','BusinessTravel','DistanceFromHome','Education', 'EducationField','Gender', 'JobLevel', 'JobRole',

       'MaritalStatus', 'MonthlyIncome', 'NumCompaniesWorked',

       'PercentSalaryHike', 'StockOptionLevel', 'TotalWorkingYears',

       'TrainingTimesLastYear', 'YearsAtCompany', 'YearsSinceLastPromotion',

       'YearsWithCurrManager']]
corr = corr_cols.corr()

plt.figure(figsize=(18,7))

sns.heatmap(corr, annot = True)

plt.show()
y = df['Attrition']

x = df.drop('Attrition', axis = 1)
from sklearn.model_selection import train_test_split

X_train,X_test, y_train, y_test = train_test_split(x,y, test_size = 0.20, random_state=42)
from sklearn.preprocessing import StandardScaler

Scaler_X = StandardScaler()

X_train = Scaler_X.fit_transform(X_train)

X_test = Scaler_X.transform(X_test)
from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression



lr = LogisticRegression()

lr.fit(X_train,y_train)

y_pred = lr.predict(X_test)



print(accuracy_score(y_test,y_pred))

print(confusion_matrix(y_test,y_pred))
from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))