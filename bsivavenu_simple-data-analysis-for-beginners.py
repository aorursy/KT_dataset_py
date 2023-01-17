# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
PATH="../input"

application_train = pd.read_csv(PATH+"/application_train.csv")
application_test = pd.read_csv(PATH+"/application_test.csv")
bureau = pd.read_csv(PATH+"/bureau.csv")
bureau_balance = pd.read_csv(PATH+"/bureau_balance.csv")
credit_card_balance = pd.read_csv(PATH+"/credit_card_balance.csv")
installments_payments = pd.read_csv(PATH+"/installments_payments.csv")
previous_application = pd.read_csv(PATH+"/previous_application.csv")
POS_CASH_balance = pd.read_csv(PATH+"/POS_CASH_balance.csv")
application_train.head()
application_test.head()
bureau.head()
bureau_balance.head()
credit_card_balance.head()
installments_payments.head()
previous_application.head()
previous_application.head()
def missing_data(data):
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data(application_train).head(10)
application_train.isnull().sum().sort_values(ascending=False)[:10]
missing_data(application_test).head(10)
missing_data(bureau)
missing_data(bureau_balance)
missing_data(credit_card_balance)
missing_data(installments_payments)
import seaborn as sns #it is my fav and very handy for beginners, plotly is though interactive but hard for beginners i believe.
import matplotlib.pyplot as plt 
sns.countplot(application_train.TARGET)
plt.show()
# TARGET value 0 means loan is repayed, value 1 means loan is not repayed.
sns.countplot(application_train.NAME_CONTRACT_TYPE.values,data=application_train)
plt.figure(figsize=(15,5))
sns.countplot(application_train.NAME_EDUCATION_TYPE.values,data=application_train)
plt.show() #to check what are the differnet categories and their count.
plt.figure(figsize=(15,5))
sns.countplot(application_train.NAME_FAMILY_STATUS.values,data=application_train)
plt.show()
plt.figure(figsize=(15,5))
sns.countplot(application_train.NAME_HOUSING_TYPE.values,data=application_train)
plt.show()
plt.figure(figsize=(15,5))
sns.countplot(application_train.NAME_INCOME_TYPE.values,data=application_train)
plt.show()
plt.figure(figsize=(15,5))
sns.countplot(application_train.NAME_TYPE_SUITE.values,data=application_train)
plt.show()
plt.figure(figsize=(15,5))
sns.countplot(application_train.NAME_INCOME_TYPE.values,data=application_train,hue=application_train.TARGET)
plt.show()
plt.figure(figsize=(15,5))
sns.countplot(application_train.NAME_EDUCATION_TYPE.values,data=application_train,hue=application_train.TARGET)
plt.show()
plt.figure(figsize=(15,5))
sns.countplot(application_train.NAME_FAMILY_STATUS.values,data=application_train,hue=application_train.TARGET)
plt.show()
plt.figure(figsize=(15,5))
sns.countplot(application_train.NAME_TYPE_SUITE.values,data=application_train,hue=application_train.TARGET)
plt.show()
plt.figure(figsize=(15,5))
sns.countplot(application_train.NAME_CONTRACT_TYPE.values,data=application_train,hue=application_train.FLAG_OWN_REALTY)
plt.show()
plt.figure(figsize=(15,5))
sns.countplot(application_train.OCCUPATION_TYPE.values,data=application_train)
plt.xticks(rotation=90)
plt.show()
plt.figure(figsize=(15,5))
sns.countplot(previous_application.NAME_CONTRACT_TYPE.values)
plt.show()
plt.figure(figsize=(15,5))
sns.countplot(previous_application.WEEKDAY_APPR_PROCESS_START.values,data=previous_application)
plt.show()
plt.figure(figsize=(15,5))
sns.countplot(previous_application.NAME_CLIENT_TYPE.values,data=previous_application)
plt.show()
