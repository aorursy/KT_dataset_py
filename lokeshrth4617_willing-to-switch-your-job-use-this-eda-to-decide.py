import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import plotly_express as px

import plotly.io as pio

pio.templates.default = "plotly_dark"

sns.set_style('darkgrid')

import pprint



import cufflinks as cf

import plotly.offline

cf.go_offline()

cf.set_config_file(offline=False, world_readable=True)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

print('Successful')
data = pd.read_csv('/kaggle/input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')

print('Data shape:',data.shape)

data.head(3)

## changing the datatype to the appropriate Datatypes

data['Attrition'] = data['Attrition'].replace('Yes',1)

data['Attrition'] = data['Attrition'].replace('No',0)
BusT = data['BusinessTravel'].value_counts()

fig = px.pie(BusT, values =BusT.values, names = BusT.index, title='Distribution of Business Travel among Employees')

fig.show()
sns.distplot(data['DailyRate'])
Dep = data['Department'].value_counts()

fig = px.pie(Dep, values = Dep.values, names = Dep.index, title='Distribution of Department')

fig.show()
sns.distplot(data['DistanceFromHome'])
ED = data['Education'].value_counts()

px.pie(ED , values = ED.values, names = ED.index)
Education = data['EducationField'].value_counts()

px.pie(Education, values = Education.values, names = Education.index)
data.drop(['EmployeeCount'], axis = 1, inplace = True)
#%matplotlib notebook

data.drop(['EmployeeNumber'], axis = 1, inplace = True)
Satisfaction = data['EnvironmentSatisfaction'].value_counts()

px.pie(Satisfaction, values = Satisfaction.values, names = Satisfaction.index)
data['Gender'] = data['Gender'].replace('Female',0)

data['Gender'] = data['Gender'].replace('Male',1)

Gender = data['Gender'].value_counts()

%matplotlib inline

sns.countplot(x = 'Gender', data =data)

print(Gender)
sns.distplot(data['HourlyRate'])
job = data['JobInvolvement'].value_counts()

px.pie(job, values = job.values, names = job.index)
data.drop(['JobLevel'], axis = 1, inplace = True)
JobRole = data['JobRole'].value_counts()

df1 = pd.DataFrame(JobRole)

df1.reset_index(inplace = True)

%matplotlib notebook

px.bar(df1 ,x = 'index', y = 'JobRole', text="JobRole")
satis = data['JobSatisfaction'].value_counts()

%matplotlib notebook

px.pie(satis, values = satis.values, names = satis.index)
data['MaritalStatus'] = data['MaritalStatus'].replace('Single',0)

data['MaritalStatus'] = data['MaritalStatus'].replace('Married',1)

data['MaritalStatus'] = data['MaritalStatus'].replace('Divorced',2)



Marital = data['MaritalStatus'].value_counts()

%matplotlib inline

df2 = pd.DataFrame(Marital)

df2.reset_index(inplace = True)

%matplotlib notebook

px.bar(df2 ,x = 'index', y = 'MaritalStatus', text="MaritalStatus")
sns.distplot(data['MonthlyIncome'])