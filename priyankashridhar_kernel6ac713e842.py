# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

%matplotlib inline

import seaborn as sn
data=pd.read_csv('/kaggle/input/loan-predication/train_u6lujuX_CVtuZ9i (1).csv')

data.head()
data.info()
data.describe()
df=data

df.head()
df.isnull().sum()
#sn.pairplot(df)
df['Loan_Status'].value_counts()
sn.countplot(df['Loan_Status'])
df['Gender'].value_counts()
df['Gender'].value_counts(normalize=True)
df['ApplicantIncome'].plot.hist(bins=20)
sn.countplot(x=df['Education'],hue=df['Loan_Status'])
df.groupby(['Gender','Education'])['Loan_Status'].count()
df.groupby(['Gender','Education'])['Loan_Status'].count().plot(kind='bar')
df['Self_Employed'].value_counts(normalize=True)
sn.countplot(df['Self_Employed'],hue=df['Loan_Status'])
#df['Loan_Amount_Term'].plot.hist(bins=20)#

sn.distplot(df['Loan_Amount_Term'])
df['Credit_History'].value_counts()
df.groupby(['Credit_History','Loan_Status'])['Loan_ID'].count()
sn.countplot(df['Credit_History'],hue=df['Loan_Status'])
df['Property_Area'].value_counts().plot(kind='bar')
df['Property_Area'].value_counts(normalize=True)
sn.countplot(df['Property_Area'],hue=df['Loan_Status'])
sn.distplot(df['CoapplicantIncome'])
fig=plt.figure(figsize=(20,10))

sn.boxplot(data = df,notch = True,linewidth = 1.5, width = 0.30)

plt.show()
df.isnull().sum()
df.dropna(inplace=True)
df.isnull().any()
from sklearn.preprocessing import LabelEncoder

var=['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']

labelencoder = LabelEncoder()

for i in var:

    df[i] = labelencoder.fit_transform(df[i])

df.dtypes    
df.head()
x=df.drop(['Loan_Status','Loan_ID'],axis=1)

y=df['Loan_Status']
from sklearn.preprocessing import OneHotEncoder

onehotencoder = OneHotEncoder(categorical_features = [0])

x = onehotencoder.fit_transform(x).toarray()
from sklearn.preprocessing import StandardScaler

scaler_ApplicantIncome = StandardScaler()

x = scaler_ApplicantIncome.fit_transform(x)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=25)
print(x_train.shape)

print(x_test.shape)

print(y_train.shape)

print(y_test.shape)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import mean_squared_error, r2_score

model= LogisticRegression()

model.fit(x_train, y_train)



Predictions = model.predict(x_test)
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test , Predictions)
from sklearn.metrics import classification_report

classification_report(y_test,Predictions)
from sklearn.metrics import accuracy_score

accuracy_score(y_test , Predictions)
x_test.shape
my_submission = pd.DataFrame({'Loan_Status':Predictions})

print(my_submission)