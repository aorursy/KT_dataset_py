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

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
df.head()
df.info()
df.isnull().sum()
import missingno as msno 

msno.matrix(df)

#distribution of males and females in different exams categories and streams

sns.catplot(x="ssc_b",hue="gender",data=df, kind="count",)

plt.ylabel("No_of_students")

plt.xlabel("senior_secondary")

sns.catplot(x="hsc_b",hue="gender",data=df, kind="count")

plt.ylabel("No_of_students")

plt.xlabel("higher_senior_secondary")

sns.catplot(x="hsc_s",hue="gender",data=df, kind="count")

plt.ylabel("No_of_students")

plt.xlabel("stream")
sns.countplot(x='workex', hue='hsc_s', data=df)
sns.countplot(x='workex', hue='hsc_b', data=df)
sns.countplot(x='workex', hue='ssc_b', data=df)
sns.kdeplot(df["salary"])

plt.show()
from sklearn.preprocessing import LabelEncoder

number = LabelEncoder()
df['gender'] = number.fit_transform(df['gender'].astype('str'))

df['degree_t'] = number.fit_transform(df['degree_t'].astype('str'))

df['specialisation'] =  number.fit_transform(df['specialisation'].astype('str'))

df['workex'] = number.fit_transform(df['workex'].astype('str'))
df.drop(['ssc_b'], axis = 1, inplace=True) 

df.drop(['hsc_s'], axis = 1, inplace=True) 

df.drop(['hsc_b'], axis = 1, inplace=True) 

df.drop(['sl_no'], axis = 1, inplace=True) 
df1 = df.copy()
df.head()
df1.head()
df = df[df["salary"]<350000.0]

df
sns.kdeplot(df["salary"])

plt.show()
df.drop(['status'], axis = 1, inplace =True) 

X1 = df.iloc[:, :9].values

y1 = df.iloc[:, 9].values

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size = 0.25, random_state = 0)
# Fitting Simple Linear Regression to the Training set

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train1, y_train1)
# Predicting the Test set results

y_pred1 = regressor.predict(X_test1)
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from math import sqrt



MAE = mean_absolute_error(y_test1, y_pred1)

print(MAE)

df1['status'] = df1['status'].map({'Not Placed':0,'Placed':1})
df1.head()
df1.drop(['salary'], axis = 1, inplace =True) 

X = df1.iloc[:, :9].values

y = df1.iloc[:, 9].values
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)

# Fitting Logistic Regression to the Training set

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0)

classifier.fit(X_train, y_train)
# Predicting the Test set results

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)

print(cm)

accuracy = accuracy_score(y_test, y_pred)*100

print(accuracy)