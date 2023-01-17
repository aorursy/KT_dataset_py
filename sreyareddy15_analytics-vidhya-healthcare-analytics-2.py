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
import matplotlib.pyplot as plt

%matplotlib inline
train = pd.read_csv('../input/av-healthcare-analytics-ii/healthcare/train_data.csv')

test = pd.read_csv('../input/av-healthcare-analytics-ii/healthcare/test_data.csv')
train.head()
train.isnull().sum()
train.describe()
train.columns
test.columns
train['Stay'].unique()
train.head(2)
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder() 

train['Department'] = label_encoder.fit_transform(train['Department'])
train['Department'] = label_encoder.fit_transform(train['Department'])

train['Department'].unique()
train['Ward_Type'] = label_encoder.fit_transform(train['Ward_Type'])

train['Ward_Type'].unique()
train['Ward_Facility_Code'] = label_encoder.fit_transform(train['Ward_Facility_Code'])

train['Ward_Facility_Code'].unique()
train['Hospital_type_code'] = label_encoder.fit_transform(train['Hospital_type_code'])

train['Hospital_type_code'].unique()
train['Hospital_region_code'] = label_encoder.fit_transform(train['Hospital_region_code'])

train['Hospital_region_code'].unique()
train['Type of Admission'] = label_encoder.fit_transform(train['Type of Admission'])

train['Type of Admission'].unique()
train['Severity of Illness'] = label_encoder.fit_transform(train['Severity of Illness'])

train['Severity of Illness'].unique()
train['Age'] = label_encoder.fit_transform(train['Age'])

train['Age'].unique()
train['Stay'] = label_encoder.fit_transform(train['Stay'])

train.Stay.unique()
train.head()
train['Bed Grade'].value_counts()
train['Bed Grade'].fillna(2.0,inplace=True)
train.isnull().sum()
train['City_Code_Patient'].value_counts()
train[train['City_Code_Patient'].isnull()]
train['City_Code_Patient'].fillna(8.0,inplace=True)
train.isnull().sum()
train.columns 
X_train= train[['Hospital_code', 'Hospital_type_code', 'City_Code_Hospital',

       'Hospital_region_code', 'Available Extra Rooms in Hospital',

       'Department', 'Ward_Type', 'Ward_Facility_Code', 'Bed Grade',

        'City_Code_Patient', 'Type of Admission',

       'Severity of Illness', 'Visitors with Patient', 'Age',

       'Admission_Deposit']]
y_train = train['Stay']

X_train.shape
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
X_train ,X_CV ,y_train,y_CV = train_test_split(X_train,y_train,test_size = 0.2,shuffle =True)
model_Logistic = LogisticRegression(solver ='sag',max_iter =1000)
model_Logistic.fit(X_train,y_train)
model.predict(X_CV)