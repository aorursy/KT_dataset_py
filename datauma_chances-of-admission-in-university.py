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
#import all the necessary libraries



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



admission = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv')
admission.head()
admission.shape
#check for null value

admission.isnull().sum()
admission.describe()
admission.info()
#delete the unncessary serial number column from the dataset.



admission.drop('Serial No.', axis=1,inplace=True)
admission.head()
#correlation matrix



fig, ax = plt.subplots(figsize=(9,9))

sns.heatmap(admission.corr(), annot=True, ax=ax, cmap='BuPu')
sns.countplot(x='Research', data=admission)

#TOEFL score



admission['TOEFL Score'].plot(kind='hist')

plt.title('Histogram of TOEFL Score')

admission['GRE Score'].plot(kind='hist')

plt.title('Histogram of GRE')
admission.plot(kind='scatter', x='TOEFL Score', y='GRE Score')

plt.title('GRE Scores Vs TOEFL score')
admission.plot(kind='scatter', x='CGPA', y='University Rating')

plt.title('University Rating Vs CGPA')
fig, ax = plt.subplots(1,2,figsize=(10,3))

admission.plot(kind='scatter', x='CGPA', y='TOEFL Score',ax=ax[0])

admission.plot(kind='scatter', x='CGPA', y='GRE Score',ax=ax[1])

plt.suptitle('CGPA VS (TOEFL Score & GRE Score)')
admission_count = admission[admission['Chance of Admit '] >=0.75]['University Rating'].value_counts()

admission_count.plot(kind='bar')

plt.title('University rating of candidates with 75% of admission chance')

plt.xlabel('University Rating')

plt.ylabel('Students count')
y = admission['Chance of Admit ']

X = admission.drop(['Chance of Admit '], axis=1)



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.2)

X_train.shape, y_train.shape, X_test.shape, y_test.shape
#Normalization



from sklearn.preprocessing import MinMaxScaler



X_scaler = MinMaxScaler(feature_range=(0,1))



X_train_norm = X_scaler.fit_transform(X_train)

X_test_norm = X_scaler.transform(X_test)
from sklearn.linear_model import LinearRegression



lr = LinearRegression()

lr.fit(X_train_norm, y_train)



y_pred = lr.predict(X_test_norm)
from sklearn.metrics import r2_score



print('R2 score ', r2_score(y_test, y_pred))
from sklearn.ensemble import RandomForestRegressor



rfg = RandomForestRegressor(n_estimators=100, random_state=0)

rfg.fit(X_train_norm, y_train)



y_pred = rfg.predict(X_test_norm)
print('R2 score ', r2_score(y_test, y_pred))
from sklearn.tree import DecisionTreeRegressor

dtr = DecisionTreeRegressor(random_state = 42)

dtr.fit(X_train_norm, y_train)



y_pred = dtr.predict(X_test_norm)
print('R2 score ', r2_score(y_test, y_pred))