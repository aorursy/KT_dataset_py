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
# Importing Required libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# Importing the dataset

df = pd.read_csv('../input/svm-classification/UniversalBank.csv')

df.head()
# Check for null values in dataset

df.isnull().mean()*100
# Assigning X and y values for the dataset

X = df.loc[:,df.columns!='CreditCard']

X.head()
y = df['CreditCard']

y.head()
# Splitting Train and Test values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None)
# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
# import svc

from sklearn.svm import SVC



# Creating a instance

classifier = SVC(kernel='rbf',random_state=None)



# Fitting the model

classifier.fit(X_train, y_train)
classifier.intercept_
classifier.n_support_
classifier.support_vectors_
# Predicting the values

y_pred_train = classifier.predict(X_train)

y_pred_test = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score



# Confusion matrix

cm_train = confusion_matrix(y_train, y_pred_train)

print(cm_train)



cm_test = confusion_matrix(y_test, y_pred_test)

print(cm_test)
# Accuracy

accuracy_train = accuracy_score(y_train, y_pred_train)

print(accuracy_train)
accuracy_test = accuracy_score(y_test, y_pred_test)

print(accuracy_test)
# Correlation

c = df.corr()

print(c)
# Heat map

plt.figure(figsize=(10,8))

sns.heatmap(c,cmap='BrBG',annot=True)
# Creditcard count

sns.countplot(df['CreditCard'])
# Credit card issue according to age

plt.figure(figsize=(10,8))

sns.countplot(df['Age'],hue=df['CreditCard'])
# Credit card issue according to Family members

sns.countplot(df['Family'],hue=df['CreditCard'])
# Credit card issue according to Education

sns.countplot(df['Education'],hue=df['CreditCard'])
# Credit card issue according to customer's Personal Loan

sns.countplot(df['Personal Loan'],hue=df['CreditCard'])
# Credit card issue according to customer's Securities Account

sns.countplot(df['Securities Account'],hue=df['CreditCard'])
# Credit card issue according to customer's CD Account

sns.countplot(df['CD Account'],hue=df['CreditCard'])