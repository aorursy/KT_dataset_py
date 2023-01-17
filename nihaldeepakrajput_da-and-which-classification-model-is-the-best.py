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
#import additional libraries
import seaborn as sns
import matplotlib.pyplot as plt
#read the dataset
df = pd.read_csv("/kaggle/input/online-shoppers-intention/online_shoppers_intention.csv")
#first few rows of the dataframe
df.head()
df.isnull().sum()
#drop the null values
df.dropna(axis = 0, how = "any", inplace = True)
#check for null values again
df.isnull().sum()
#describe the dataframe
df.describe().T
#Count of Revenue
sns.countplot(df.Revenue, palette = 'seismic_r')
#pie chart for revenue
labels = ['False', 'True']
plt.title("Revenue")
plt.pie(df.Revenue.value_counts(), labels = labels, autopct = '%.4f%%')
plt.legend()
#different users
sns.countplot(x = df.VisitorType)
#different regions
sns.countplot(df.Region)
#Operating system wrt Revenue
sns.countplot(df.OperatingSystems, hue = df.Revenue)
#Revenue for each region
sns.countplot(df.Region, hue = df.Revenue)
#Revenue with respect to Weekend
sns.countplot(df.Weekend, hue = df.Revenue)
plt.figure(figsize = (10,10))
sns.heatmap(df.corr())
#check datatypes
df.dtypes
#split into X and y
X = df.iloc[:,6:-1].values
y = df.iloc[:,-1].values
#label encode the objects
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
y = le.fit_transform(y)
X[:,4] = le.fit_transform(X[:,4])
X[:,9] = le.fit_transform(X[:,9])
X[:,10] = le.fit_transform(X[:,10])
#split into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(solver = 'lbfgs')
classifier.fit(X_train, y_train)
#predict the values
y_pred = classifier.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(metric = 'minkowski', p = 2, n_neighbors = 5)
classifier.fit(X_train, y_train)
#predict the values
y_pred = classifier.predict(X_test)
#print report
print(classification_report(y_test, y_pred))
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)
#predict the values
y_pred = classifier.predict(X_test)
#print report
print(classification_report(y_test, y_pred))
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
#predict the values
y_pred = classifier.predict(X_test)
#print report
print(classification_report(y_test, y_pred))
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
#predict the values
y_pred = classifier.predict(X_test)
#print report
print(classification_report(y_test, y_pred))