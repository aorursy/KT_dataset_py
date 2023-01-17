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
# Importing some additional libraries

import matplotlib.pyplot as plt

import seaborn as sns



# models that I want to try with this dataset to predict the churning customers

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier



# some utility libs

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



import warnings

warnings.filterwarnings("ignore")
df = pd.read_csv('/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')

display(df.head())

display(df.columns)

display(df.describe())

display(df.info())
for col in df.columns:

    # excluding the numerical columns

    if col not in ['customerID','MonthlyCharges', 'TotalCharges', 'tenure']:

        print(col,':', df[col].unique())
# 'No ineternet service' & 'No phone service' will probably need to be removed since they are indicated in their own columns

# As a rough check, let's see how churn and non-churn customers are distributed by each categorical columns



for col in df.columns:

    if col not in ['customerID','MonthlyCharges', 'TotalCharges', 'tenure']:

        print(col,':', df[col].unique(),'\n')

        sns.countplot(col, data=df, palette='coolwarm', hue='Churn')

        plt.show()
# creating a copy to clean the data and store

df_edited = df.copy()



# cleaning up the categorical into booleans

df_edited = df.replace({'No internet service':0, 'No phone service':0, 'No':0, 'Yes':1})

df_edited.replace({'InternetService': {0: 'No'}}, inplace=True)



# Replacing Empty Total Charges with Monthly Charges under the assumption that these are new accounts & converting it to float

df_edited['TotalCharges'].loc[df_edited['TotalCharges']==' '] = df_edited[df_edited['TotalCharges']==' ']['MonthlyCharges']

df_edited['TotalCharges'] = df_edited.TotalCharges.astype('float64')



# dropping the customer id since it is very unlikeyly that it's useful

df_edited.drop('customerID', axis=1, inplace=True)



# getting dummy values for the categorical values

gender = pd.get_dummies(df_edited.gender, drop_first=True, prefix='gender')

PaymentMethod = pd.get_dummies(df_edited.PaymentMethod, drop_first=False, prefix='PaymentMethod')

Contract = pd.get_dummies(df_edited.Contract, drop_first=False, prefix='Contract')

InternetService = pd.get_dummies(df_edited.InternetService, drop_first=False, prefix='InternetService')



# drop the original columns and add dummy columns

df_edited.drop(['gender', 'PaymentMethod','Contract','InternetService'], axis=1, inplace=True)

df_edited = pd.concat([df_edited, gender, PaymentMethod, Contract, InternetService], axis=1)



# let's check if the columns are clean enough to be used in modeling

for col in df_edited.columns:

    if col not in ['customerID','MonthlyCharges', 'TotalCharges', 'tenure']:

        print(col,':', df_edited[col].unique(),'\n')
for col in ['MonthlyCharges', 'TotalCharges', 'tenure']:

    sns.distplot(df_edited[col])

    plt.show()
g = sns.lmplot(x='tenure', y='TotalCharges', data=df_edited, col = 'Churn',

          scatter_kws = dict(s=1))
model_linear = LinearRegression()

no_churn = df_edited[df_edited['Churn']==0]

churn = df_edited[df_edited['Churn']==1]

model_linear.fit(X=no_churn['tenure'].values.reshape(-1,1),y=no_churn['TotalCharges'].values.reshape(-1,1))

print('no churn:',model_linear.coef_[0][0])

model_linear.fit(X=churn['tenure'].values.reshape(-1,1),y=churn['TotalCharges'].values.reshape(-1,1))

print('churn:',model_linear.coef_[0][0])
Churn = df_edited[df_edited.Churn==1]

No_Churn = df_edited[df_edited.Churn==0]



sns.distplot(Churn[['MonthlyCharges']], hist=False, rug=True, label='Churn', axlabel='MonthlyCharges')

sns.distplot(No_Churn[['MonthlyCharges']], hist=False, rug=True, label='No Churn')

plt.show()
print(df_edited.Churn.value_counts())
# we will use randomized 1869 customers of no_churn group.

df_edited = df_edited.sample(frac=1)

df_edited_balanced = pd.concat([df_edited[df_edited['Churn']==1],df_edited[df_edited['Churn']==0][:1869]],axis=0)

sns.countplot(df_edited_balanced['Churn'], palette='coolwarm')

print(df_edited_balanced.Churn.value_counts())
# Prepare the train and test data sets

X=df_edited_balanced.drop('Churn', axis=1)

y=df_edited_balanced.Churn

test_size = 0.3

random_state = 70

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)



models = [LogisticRegression(),DecisionTreeClassifier(), RandomForestClassifier(), XGBClassifier()]

for model in models:

    print ('\n','-'*100,'\n',model)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(confusion_matrix(y_test,y_pred))

    print(classification_report(y_test,y_pred))
# Prepare the train and test data sets

X=df_edited.drop('Churn', axis=1)

y=df_edited.Churn

test_size = 0.3

random_state = 70

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)



models = [LogisticRegression(),DecisionTreeClassifier(), RandomForestClassifier(), XGBClassifier()]

for model in models:

    print ('\n','-'*100,'\n',model)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(confusion_matrix(y_test,y_pred))

    print(classification_report(y_test,y_pred))