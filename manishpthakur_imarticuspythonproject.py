import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('seaborn')

#importing data
data1=pd.read_csv("../input/PythonProject.csv",low_memory=False)
data1.head()
data1.shape
print(data1.isnull().sum()/data1.shape[0]*100)
data1=data1.loc[:, data1.isnull().sum() < 0.8*data1.shape[0]]
data1.shape
data1.describe(include='all')
h={}
ctr1=0
ctr2=0
for i in range(data1.shape[1]):
    if isinstance(data1[data1.columns[i]].dropna().iloc[1],str):
        h[data1.columns[i]]=data1[data1.columns[i]].mode()[0]
        ctr1=ctr1+1
    else:
        h[data1.columns[i]]=data1[data1.columns[1]].median()
        ctr2=ctr2+1
print(ctr1)
print(ctr2)
data1.fillna(h,inplace=True)
data1.head()
print(data1.shape)
data1.describe(include='all')
print(data1.isnull().sum()/data1.shape[0]*100,"%")
#finding cat variables in data1
data2=pd.DataFrame()
for x in data1.columns:
    if data1[x].dtype=='object':
        data2[x]=data1[x]
        print(x)
#data2.columns
colname=['term', 'grade', 'sub_grade', 'emp_title', 'emp_length',
       'home_ownership', 'verification_status', 'issue_d', 'pymnt_plan',
       'purpose', 'title', 'zip_code', 'addr_state', 'earliest_cr_line',
       'initial_list_status', 'last_pymnt_d', 'next_pymnt_d',
       'last_credit_pull_d', 'application_type']
 #for preprocessing the data
from sklearn import preprocessing
le={}       
for x in colname:
    le[x]=preprocessing.LabelEncoder()
for x in colname:
    data1[x]=le[x].fit_transform(data1[x])
#finding cat variables in data1, there qill be none.
for x in data1.columns:
    if data1[x].dtype=='object':
        print(x)
for x in data1.columns:
    print(data1[x].dtype)
data1.head()