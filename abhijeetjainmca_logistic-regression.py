# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.





df_emp_data = pd.read_csv('../input/hr-analytics-case-study/general_data.csv')

df_emp_survey = pd.read_csv('../input/hr-analytics-case-study/employee_survey_data.csv')



df_emp_data.head()
df_emp_data.info()
df_emp_data.TotalWorkingYears.fillna(0,inplace=True)

df_emp_data.NumCompaniesWorked.fillna(0,inplace=True)

df_emp_data.info()
df_emp_survey.info()
df_emp_survey.isna().sum()
df_emp_survey['EnvironmentSatisfaction'].fillna(df_emp_survey['EnvironmentSatisfaction'].mode()[0],inplace=True)

df_emp_survey['JobSatisfaction'].fillna(df_emp_survey['JobSatisfaction'].mode()[0],inplace=True)

df_emp_survey['WorkLifeBalance'].fillna(df_emp_survey['WorkLifeBalance'].mode()[0],inplace=True)

df_emp_survey.info()
df_emp_data = pd.merge(df_emp_data,df_emp_survey,on='EmployeeID')

df_emp_data.head()
sns.countplot(data=df_emp_data,x='BusinessTravel',hue='Gender')
sns.countplot(data=df_emp_data,x='Attrition',hue='Gender')
sns.countplot(data=df_emp_data,x='Department',hue='Attrition')
plt.figure(figsize=(12,6))

sns.countplot(data=df_emp_data,x='EducationField',hue='Attrition')
final_working_ds = df_emp_data.drop(['EmployeeCount','Over18','StandardHours','EmployeeID'],axis=1)



final_working_ds.info()
def impute_age(age):

    if (age>15 & age<=30):

        return 1

    elif (age>30 & age<=45):

        return 2

    elif (age>45 & age<=60):

        return 3



final_working_ds['Age']=final_working_ds.Age.apply(impute_age)



final_working_ds.Gender.value_counts()
gender_dummy = pd.get_dummies(final_working_ds['Gender'],drop_first=True)



#Concatinating the newly created dummy of Gender into Dataset



final_working_ds = pd.concat([final_working_ds,gender_dummy],axis=1)



#Dropping the existed column of Gender as Dummy variable is added



final_working_ds.drop(['Gender'],inplace=True,axis=1)



final_working_ds.head()
from sklearn.preprocessing import LabelEncoder

LabelE_BusinessTravel = LabelEncoder()

final_working_ds['BusinessTravel'] = LabelE_BusinessTravel.fit_transform(final_working_ds['BusinessTravel'])

LabelE_Department = LabelEncoder()

final_working_ds['Department'] = LabelE_Department.fit_transform(final_working_ds['Department'])

LabelE_EducationField = LabelEncoder()

final_working_ds['EducationField'] = LabelE_EducationField.fit_transform(final_working_ds['EducationField'])

LabelE_JobRole = LabelEncoder()

final_working_ds['JobRole'] = LabelE_JobRole.fit_transform(final_working_ds['JobRole'])

LabelE_MaritalStatus = LabelEncoder()

final_working_ds['MaritalStatus'] = LabelE_MaritalStatus.fit_transform(final_working_ds['MaritalStatus'])

LabelE_Attrition = LabelEncoder()

final_working_ds['Attrition'] = LabelE_Attrition.fit_transform(final_working_ds['Attrition'])



final_working_ds.head()
from sklearn.preprocessing import StandardScaler

scalr = StandardScaler()

scalr.fit(final_working_ds.drop('Attrition',axis=1))

scaled_features = scalr.transform(final_working_ds.drop('Attrition',axis=1))
colms = final_working_ds.columns.delete(1)

X = pd.DataFrame(scaled_features,columns=colms)

y = final_working_ds['Attrition']
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=101)
from sklearn.linear_model import LogisticRegression

log_Reg = LogisticRegression()



log_Reg.fit(X_train,y_train)



y_pred = log_Reg.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix



print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))