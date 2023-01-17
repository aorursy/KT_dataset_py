# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/employee-attrition/employee_attrition_train.csv')

df.head()
df.info()
categorical = [x for x in df.columns if df[x].dtype == 'object']

numerical = [x for x in df.columns if df[x].dtype != 'object']

print('There are {} categorical features in the dataset'.format(len(categorical)),'\n')

for i in categorical:

    print(i)

print('\n')

print('There are {} numerical features in the dataset'.format(len(numerical)),'\n')

for i in numerical:

    print(i)
# Checking for missing values

print('Missing values in the dataset','\n')

print(df.isnull().sum().sort_values(ascending=False),'\n')

print('The total missing values are {} and it is {}% of the dataset'

      .format(df.isnull().sum().sum(),round((df.isnull().sum().sum())*100/df.shape[0],3)))
df.duplicated().sum()
df.describe(include='all').T
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan,strategy='median')

df[['Age','DistanceFromHome','DailyRate']] = imputer.fit_transform(df[['Age','DistanceFromHome','DailyRate']])

imputer2 = SimpleImputer(missing_values=np.nan,strategy='most_frequent')

df[['BusinessTravel','MaritalStatus']] = imputer2.fit_transform(df[['BusinessTravel','MaritalStatus']])

imputer3 = SimpleImputer(missing_values=0,strategy='median')

df[['YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion','YearsWithCurrManager','StockOptionLevel','TotalWorkingYears', 'TrainingTimesLastYear','NumCompaniesWorked']] = imputer3.fit_transform(df[['YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion','YearsWithCurrManager','StockOptionLevel','TotalWorkingYears', 'TrainingTimesLastYear','NumCompaniesWorked']])
df.isnull().sum().sum()
df.hist(figsize=(20,20))
plt.figure(figsize=(10,15))

sns.boxplot(data=df,orient='h')
df = df.drop(['Over18','EmployeeCount','StandardHours'],axis=1)
jobs =pd.crosstab(index=df['EducationField'],columns=df['JobRole'],values=df['EducationField'],aggfunc='count')

jobs
fig , ax = plt.subplots(nrows=4,ncols=2)

fig.set_size_inches(25,20)

sns.barplot(jobs['Healthcare Representative'],jobs.index,ax=ax[0][0])

sns.barplot(jobs['Human Resources'],jobs.index,ax=ax[0][1])

sns.barplot(jobs['Laboratory Technician'],jobs.index,ax=ax[1][0])

sns.barplot(jobs['Manager'],jobs.index,ax=ax[1][1])

sns.barplot(jobs['Manufacturing Director'],jobs.index,ax=ax[2][0])

sns.barplot(jobs['Research Director'],jobs.index,ax=ax[2][1])

sns.barplot(jobs['Research Scientist'],jobs.index,ax=ax[3][0])

sns.barplot(jobs['Sales Executive'],jobs.index,ax=ax[3][1])
sns.barplot(jobs['Sales Representative'],jobs.index)
plt.figure(figsize=(10,15))

sns.boxplot(data=df,orient='h')
plt.figure(figsize=(22,6))

sns.boxplot(data=df[['Age','NumCompaniesWorked','PerformanceRating','StockOptionLevel','TotalWorkingYears','TrainingTimesLastYear','YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion','YearsWithCurrManager']],orient='v')
def outliers(column):

    q1 = np.quantile(column,0.25)

    q3 = np.quantile(column,0.75)

    iqr = q3-q1

    lrlimit = q1 - (1.5*iqr)

    urlimit = q3 + (1.5*iqr)

    return lrlimit, urlimit
cols = ['Age','NumCompaniesWorked','PerformanceRating','StockOptionLevel','TotalWorkingYears','TrainingTimesLastYear','YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion','YearsWithCurrManager']

for i in cols:

    if df[i].dtype != 'object':

        print(i , outliers(df[i]),'\n')

    
for i in cols:

    lrlimit , urlimit = outliers(df[i])

    df[i] = np.where(df[i]>urlimit, urlimit,df[i])

    df[i] = np.where(df[i]<lrlimit, lrlimit,df[i])
plt.figure(figsize=(22,6))

sns.boxplot(data=df[['Age','NumCompaniesWorked','PerformanceRating','StockOptionLevel','TotalWorkingYears','TrainingTimesLastYear','YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion','YearsWithCurrManager']],orient='v')
fig , ax = plt.subplots(8)

fig.set_size_inches(20,50)

d=0

for i in df.columns:

    if df[i].dtype == 'object':

        print(i ,'\n', df[i].value_counts(),'\n')

        df[i].value_counts().plot(kind='pie',ax=ax[d])

        d = d+1
df = pd.get_dummies(data=df,columns = ['Gender','MaritalStatus','OverTime','Department','EducationField','JobRole'],drop_first=True)

df.head()
travel = {'Non-Travel' : 0 ,'Travel_Rarely' : 1,'Travel_Frequently' : 2}
df['Attrition'] = pd.Categorical(df['Attrition']).codes

df['BusinessTravel'] = df['BusinessTravel'].replace(travel)

df.head()
X = df.drop('Attrition',axis=1)



Y = df['Attrition']
from sklearn.model_selection import train_test_split
# Splitting the data into training and test data

X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size=0.3,random_state = 1)
from sklearn.naive_bayes import GaussianNB
# Importing the model and fitting the training data

model = GaussianNB()

model.fit(X_train , Y_train)
# Accuracy score for the training data

model.score(X_train,Y_train)
# Accuracy score for the test data

model.score(X_test,Y_test)
# Predicting the Attrition for training and test data

pred_train = model.predict(X_train)

pred_test = model.predict(X_test)
from sklearn.metrics import confusion_matrix , classification_report
# Confusion Matrix for training data

print(confusion_matrix(Y_train,pred_train))

print(classification_report(Y_train,pred_train))
# Confusion Matrix for test data

print(confusion_matrix(Y_test,pred_test))

print(classification_report(Y_test,pred_test))