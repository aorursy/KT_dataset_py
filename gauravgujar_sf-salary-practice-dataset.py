# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#importing libraries which we required further

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
#lets read the salary data and store it in dataframe using pandas

data = pd.read_csv('/kaggle/input/sf-salaries/Salaries.csv')

data.head()
#lets see overview of dataset using describe()

data.describe()
data['Benefits'].value_counts()
data['BasePay'].value_counts()
data['Status'].value_counts()
#since there are lot of missing values in status and notes variable in datsaet so lets drop it 

data.drop(columns=["Status","Notes"],inplace=True,axis=1)
data.head()
#lets identify categorical columns in dataset

# Get list of categorical variables

s = (data.dtypes == 'object')

object_cols = list(s[s].index)



print("Categorical variables:")

print(object_cols)
data.info()
data.isnull().sum()
for col in ['BasePay','OvertimePay','OtherPay','Benefits']:

    data[col]=pd.to_numeric(data[col],errors='coerce')
#data.fillna(value='BasePay')

"""data['BasePay']=data.BasePay.fillna(data['BasePay'].mean(),inplace=True)

data['Benefits']=data.Benefits.fillna(data['Benefits'].mean(),inplace=True)"""
data.isnull().sum()
data['JobTitle'].value_counts()
print(data.JobTitle.unique())
data['EmployeeName'] = data['EmployeeName'].apply(str.upper)

data.head()
data['JobTitle'] = data['JobTitle'].apply(str.upper)

data['JobTitle'].value_counts()
d_hsp={"1":"I","2":"II","3":"III","4":"IV","5":"V","6":"VI","7":"VII","8":"VIII",

       "9":"IX","10":"X","11":"XI","12":"XII","13":"XIII","14":"XIV","15":"XV",

       "16":"XVI","17":"XVII","18":"XVIII","19":"XIX","20":"XX","21":"XXI",

       "22":"XXII","23":"XXIII","24":"XXIV","25":"XXV"}

data['JobTitle'] = data['JobTitle'].replace(d_hsp, regex=True)
data['JobTitle'].value_counts()