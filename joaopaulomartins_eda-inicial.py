# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



sns.set(rc={'figure.figsize':(11.7,8.27)})





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data= pd.read_csv('../input/train.csv')



data.head()
data['gender'] = data['gender'].apply(lambda x: 1 if x=='Male' else 0)



data.head()
data['Partner'] = data['Partner'].apply(lambda x: 1 if x == 'Yes' else 0)

data['Dependents'] = data['Dependents'].apply(lambda x :1 if x == 'Yes' else 0)

data['PhoneService'] = data['PhoneService'].apply(lambda x :1 if x == 'Yes' else 0)



data.head()
#Valores únicos da coluna

data['MultipleLines'].unique()
#Converte a coluna para one hot enconding

data = pd.get_dummies(data, columns=['MultipleLines'])



data.head()
sns.countplot(data['InternetService'])
data = pd.get_dummies(data, columns=['InternetService'])



data.head()
sns.countplot(data['OnlineSecurity'])
data = pd.get_dummies(data, columns=['OnlineSecurity'])

data.head()
data = pd.get_dummies(data, columns=['OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'])



data.head()
sns.countplot(data['Contract'])
data = pd.get_dummies(data, columns=['Contract'])

data.head()
sns.countplot(data['PaperlessBilling'])
data['PaperlessBilling'] = data['PaperlessBilling'].apply(lambda x: 1 if x == 'Yes' else 0)

data.head()
sns.countplot(data['PaymentMethod'])
data = pd.get_dummies(data, columns=['PaymentMethod'])

data.head()
sns.distplot(data['tenure'])
sns.distplot(data['MonthlyCharges'])
sns.distplot(data['TotalCharges'])
corr = data.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)
data.to_csv('initial_data.csv', index=False)