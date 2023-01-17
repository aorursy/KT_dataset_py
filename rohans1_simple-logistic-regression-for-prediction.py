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
os.listdir('../input/telco-customer-churn')
import numpy as np

import pandas as pd

import pandas_profiling as pdf

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

from pylab import rcParams

%matplotlib inline

#loading the CSV

filepath = "../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv"

data = pd.read_csv(filepath)
data.head()
data.shape
#plotting the data

sizes = data['Churn'].value_counts(sort=True)

print(sizes)

colors = ['orange', 'red']

#how much and which portion to "explode out of chart

explode=[0.0, 0.1]

rcParams['figure.figsize'] = 5,5

labels =['no', 'yes']



#plot 

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=270,)



plt.title('percentage of churn in dataset')

plt.show()
data.drop(['customerID'], axis=1, inplace=True)
data.isnull().sum()
data.dtypes
pdf.ProfileReport(data)
data.describe()
data['TotalCharges'] = data['TotalCharges'].replace(" ", 0).astype('float32')
data.isnull().sum()
data['TotalCharges']
data['Churn'].replace(to_replace='Yes', value=1, inplace=True)

data['Churn'].replace(to_replace='No', value=0, inplace=True)
data.gender = [1 if each == 'Male' else 0 for each in data.gender]

columns_to_convert = ['Partner',

                      'Dependents',

                      'PhoneService',

                      'OnlineSecurity',

                      'OnlineBackup',

                      'DeviceProtection',

                      'TechSupport',

                      'StreamingTV',

                      'StreamingMovies',

                      'PaperlessBilling']

for item in columns_to_convert:

    data[item] = [1 if each == 'Yes' else 0 for each in data[item]]

data.head()
import seaborn as sns

g1 = sns.catplot(x='Contract', y='Churn', data=data, kind='bar')

g1.set_ylabels('Churn Probability')



g2= sns.catplot(x='InternetService', y='Churn', data=data, kind='bar')

g2.set_ylabels('Churn Probability')
data = pd.get_dummies(data)

data.dtypes
data.head()
data.corr()['Churn'].sort_values()
Y = data['Churn'].values

X = data.drop(labels =['Churn'], axis=1)



#create train and test dataset

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=101)

#importing logistic regerssion model

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

result = model.fit(X_train, y_train)
from sklearn import metrics

prediction = model.predict(X_test)

#print the prdiction accuracy

print(metrics.accuracy_score(y_test, prediction))
from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

results = confusion_matrix(y_test, prediction)

print('Confusion Matrix: ')

print(results)

print(classification_report(y_test, prediction))