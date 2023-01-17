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
import numpy as np
import pandas as pd

df = pd.read_csv('../input/loan-predication/train_u6lujuX_CVtuZ9i (1).csv')
df.head()
df.shape
df.isnull().sum()
df.dtypes ## by using .dtypes we will get the data type of each column
import seaborn as sns
sns.countplot('Loan_Status',hue='Gender',data=df)
df['Gender'] = df['Gender'].fillna('Male')
sns.countplot('Loan_Status',hue='Married',data=df)
df['Married'] = df['Married'].fillna('Yes')
sns.countplot('Loan_Status',hue='Dependents',data=df)
df['Dependents'] = df['Dependents'].fillna('0')
sns.countplot('Loan_Status' , hue = 'Self_Employed' , data = df)
df['Self_Employed']=df['Self_Employed'].fillna('No')
sns.distplot(df['LoanAmount'])
sns.scatterplot(df['LoanAmount'],y=np.arange(0,614))
mean=df[df['LoanAmount']<=400]['LoanAmount'].mean()
df['LoanAmount'].fillna(mean,inplace=True)
sns.scatterplot(df['Loan_Amount_Term'],y=np.arange(0,614))
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].value_counts().idxmax(), inplace=True)
df['Credit_History'].unique()
sns.countplot('Loan_Status',hue='Credit_History',data=df)
df['Credit_History'].fillna(df['Credit_History'].value_counts().idxmax(), inplace=True)
df.isnull().sum()
df.head(10)
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
df_data = df.copy()
print(df_data['Gender'].unique())
print(df_data['Married'].unique())
print(df_data['Education'].unique())
print(df_data['Self_Employed'].unique())
print(df_data['Property_Area'].unique())
df_data['Gender'] = lb.fit_transform(df_data['Gender'])
df_data['Married'] = lb.fit_transform(df_data['Married'])
df_data['Education'] = lb.fit_transform(df_data['Education'])
df_data['Self_Employed'] = lb.fit_transform(df_data['Self_Employed'])
df_data['Property_Area'] = lb.fit_transform(df_data['Property_Area'])
df_data['Loan_Status'] = lb.fit_transform(df_data['Loan_Status'])
df_data['Dependents'] = lb.fit_transform(df_data['Dependents'])
print(df_data['Gender'].unique())
print(df_data['Married'].unique())
print(df_data['Education'].unique())
print(df_data['Self_Employed'].unique())
print(df_data['Property_Area'].unique())
print(df_data['Loan_Status'].unique())
print(df_data['Dependents'].unique())
df_data.head()
df_data.columns
X = df_data.drop(['Loan_ID','Loan_Status'],axis=1)
Y = df_data[['Loan_Status']]
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.3,random_state=43)
from sklearn.preprocessing import StandardScaler
std = StandardScaler()
std.fit(xtrain)
xtr = std.transform(xtrain)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
lgt = LogisticRegression()

lgt.fit(xtr,ytrain)

xts = std.transform(xtest)
predict = lgt.predict(xts)
print(predict)

print(classification_report(ytest,predict))