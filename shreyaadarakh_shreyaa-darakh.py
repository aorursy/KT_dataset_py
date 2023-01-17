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
train = pd.read_csv("/kaggle/input/credit-risk-modeling-case-study/CRM_TrainData.csv")
test = pd.read_csv("/kaggle/input/credit-risk-modeling-case-study/CRM_TestData.csv")
print(len(train))
print(len(test))
train = train.drop('Loan ID',1)

train = train.drop('Customer ID',1)
test= test.drop('Customer ID',1)

test= test.drop('Loan ID',1)
def cleanTheData(pd_original_data):

    data_cleaning = pd_original_data

    

    con_dict = {'Current Loan Amount': float} 

    data_cleaning = data_cleaning.astype(con_dict)

  

    data_cleaning['Credit Score'].fillna(data_cleaning['Credit Score'].mean(),inplace=True)

   

    data_cleaning['Annual Income'].fillna(data_cleaning['Annual Income'].median(),inplace=True)

  

    data_cleaning["Months since last delinquent"].fillna("0",inplace=True)

    con_dict = {'Months since last delinquent': int} 

    data_cleaning = data_cleaning.astype(con_dict)

    

    mode = data_cleaning['Years in current job'].mode()

    data_cleaning['Years in current job'].replace('[^0-9]',"",inplace=True,regex=True)

    data_cleaning['Years in current job'] = data_cleaning['Years in current job'].fillna(10)

    con_dict = {'Years in current job': int} 

    data_cleaning = data_cleaning.astype(con_dict)

   

    data_cleaning["Maximum Open Credit"].replace('[a-zA-Z@_!#$%^&*()<>?/\|}{~:]',"0",regex=True,inplace=True)

    con_dict = {'Maximum Open Credit': float} 

    data_cleaning = data_cleaning.astype(con_dict)

  

    data_cleaning[data_cleaning.Bankruptcies.isna()==True]

    data_cleaning.Bankruptcies.fillna(0.0,inplace=True)

    con_dict = {'Bankruptcies': int} 

    data_cleaning = data_cleaning.astype(con_dict)

    

    data_cleaning['Tax Liens'].fillna(0.0,inplace=True)

    con_dict = {'Tax Liens': int} 

    data_cleaning = data_cleaning.astype(con_dict)

    

    con_dict = {'Monthly Debt': float} 

    data_cleaning["Monthly Debt"].replace('[^0-9.]',"",regex=True,inplace=True )

    data_cleaning["Monthly Debt"] = data_cleaning["Monthly Debt"].astype(con_dict)

    

    return data_cleaning
train = cleanTheData(train)

test = cleanTheData(test)
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
y = train["Loan Status"]

x = train.drop("Loan Status",1)
x_cate = train.loc[:,('Term','Home Ownership','Purpose','Years in current job','Number of Open Accounts')]

x_cate_test = test.loc[:,('Term','Home Ownership','Purpose','Years in current job','Number of Open Accounts')]

x_rem = train.drop(columns = ['Loan Status','Term','Home Ownership','Purpose','Years in current job','Number of Open Accounts'])
print(len(x_rem))
x_rem_test = test.drop(columns = ['Term','Home Ownership','Purpose','Years in current job','Number of Open Accounts'])

print(len(x_rem_test))
x_rem_test.head()
x_rem_test = x_rem_test.drop('Unnamed: 2',1)
labels = list(x_rem)

mm = MinMaxScaler()

x_scaled = pd.DataFrame(mm.fit_transform(x_rem), columns=labels)
labels = list(x_rem_test)

mm = MinMaxScaler()

x_scaled_test = pd.DataFrame(mm.fit_transform(x_rem_test), columns=labels)
x_cate_onehot = pd.get_dummies(x_cate)

x_cate_onehot_test = pd.get_dummies(x_cate_test)

x_cate_onehot_test.isnull().sum()
X = pd.concat([x_scaled, x_cate_onehot], axis = 1)
x_rem_test.isnull().sum()
X_test = pd.concat([x_rem_test, x_cate_onehot_test], axis = 1)
print(len(x_rem_test), len(x_cate_onehot_test))
X_test.isnull().sum()
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

model = lr.fit(X,y)
y_pred = lr.predict(X_test)
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC as SVM

svm = SVM()

svm.fit(X, y)
y_pred = svm.predict(X_test)
y_pred
attempt = pd.read_csv("/kaggle/input/credit-risk-modeling-case-study/CRM_TestData.csv")
x1 = attempt['Loan ID']
sub_csv = pd.DataFrame({'Loan ID':x1,'Loan Status':y_pred})

sub_csv.to_csv('sub3.csv')