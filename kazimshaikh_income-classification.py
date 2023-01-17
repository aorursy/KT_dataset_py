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
# Getting the CSV file in 'SalData' and creating a DataFrame for the same

SalData = pd.read_csv("/kaggle/input/income/train.csv")

SalData = pd.DataFrame(SalData, columns = SalData.columns )

SalData
# Checking the info() which will give a quick glance to our Dataset

SalData.info()
#Checking for Null Values in DataFrame

SalData.isnull().values.any() 
#Checking which specific column has the null value

SalData.isnull().sum()
# Changing the column name to 'predclass' and furthur deleting the previous column

SalData['predclass'] = SalData['income_>50K']

del SalData['income_>50K']
# Printing all the Unique values from the columns

print('workclass',SalData.workclass.unique())

print('education',SalData.education.unique())

print('marital-status',SalData['marital-status'].unique())

print('occupation',SalData.occupation.unique())

print('relationship',SalData.relationship.unique())

print('race',SalData.race.unique())

print('gender',SalData.gender.unique())

print('native-country',SalData['native-country'].unique())
import seaborn as sns

import matplotlib.pyplot as plt
# Plotting the 'predclass'

fig = plt.figure(figsize=(20,1))

plt.style.use('seaborn-ticks')

sns.countplot(y="predclass", data=SalData)



# Where 0 is Salary<50K and 1 is Salary>50K

# Here we can see income level less than 50K is more than 3 times of those above 50K
SalData[['education', 'educational-num']].groupby(['education'], as_index=False).mean().sort_values(by='educational-num', ascending=False)
# Plotting 'education' column

fig = plt.figure(figsize=(20,3))

plt.style.use('seaborn-ticks')

sns.countplot(y="education", data=SalData)
# Plotting 'marrital-status' column

fig = plt.figure(figsize=(20,2))

plt.style.use('seaborn-ticks')

sns.countplot(y="marital-status", data=SalData)
# Plotting 'occupation' column

plt.style.use('seaborn-ticks')

plt.figure(figsize=(20,4)) 

sns.countplot(y="occupation", data=SalData)
# Plotting 'workclass' column

plt.style.use('seaborn-ticks')

plt.figure(figsize=(20,3)) 

sns.countplot(y="workclass", data=SalData)
# One Hot Encoding

OHEdata = pd.get_dummies(SalData)
# Here you can see the number of column has increased from 16 to 106 

# to know more visit: https://www.geeksforgeeks.org/ml-one-hot-encoding-of-datasets-in-python/

OHEdata.columns
# Train Test Split the Data

X = OHEdata.drop(['predclass'], axis = 1)
y = OHEdata['predclass']

y.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 5)
X_train
X_test
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train_sc = sc.fit_transform(X_train)

X_test_sc = sc.transform(X_test)
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score





DT_classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 1219)

DT_classifier.fit(X_train, y_train)

y_pred_DT = DT_classifier.predict(X_test)

accuracy_score(y_test, y_pred_DT)
from sklearn.neighbors import KNeighborsClassifier



knn_classifier = KNeighborsClassifier(n_neighbors = 7, metric = 'minkowski', p = 5)

knn_classifier.fit(X_train, y_train)

y_pred_knn = knn_classifier.predict(X_test)

accuracy_score(y_test, y_pred_knn)