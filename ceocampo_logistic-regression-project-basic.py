# Import relevant libraries

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

%matplotlib inline
# Load data

df = pd.read_csv('../input/advertising/advertising.csv')

df.head()
# Data contains a list of users on a particular website. Data tracks if the potential customer clicked on an ad or not and contains various features of these potential customers.

# Features include daily time spent on site, age, income, sex, country, time of usage, etc.

# Goal is to use this data to predict if a potential user on this website will click an ad or not.

# While other models such as SVM, random forest, XGBoost, etc. can be used for this classification problem, linear regression was used for demonstration.
df.info()
df.describe()
# Check for missing values

df.isnull().sum()
sns.distplot(df['Age'], kde=False, bins=30)

plt.show()
sns.jointplot(x='Age', y='Area Income', data=df)

plt.show()
sns.jointplot(x='Age', y='Daily Time Spent on Site', data=df, kind='kde')

plt.show()
sns.jointplot(x='Daily Time Spent on Site', y='Daily Internet Usage', data=df)

plt.show()
sns.pairplot(df, hue='Clicked on Ad')

plt.show()
df.columns
df.shape
# Convert values under 'Timestamp' from str to datetime

df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Timestamp'][0]
# Creating new columns for 'Month', 'Day of Week', and 'Hour' from information in 'Timestamp' after converting to a datetime

df['Month'] = df['Timestamp'].apply(lambda t:t.month)

df['Day of Week'] = df['Timestamp'].apply(lambda t:t.dayofweek)

df['Hour'] = df['Timestamp'].apply(lambda t:t.hour)
df.head()
# Will convert to dummy variables

print(df['Country'].nunique())



# Nearly all rows in this column are unique. Best to drop entire column prior to fitting the model.

print(df['City'].nunique())



# Each row in this column is unique. Best to drop entire column prior to fitting the model.

print(df['Ad Topic Line'].nunique())



# Each row in this column is unique. Best to drop entire column prior to fitting the model.

print(df['Ad Topic Line'].nunique())
# Dropping colunms that will be excluded

df.drop(['Ad Topic Line', 'City', 'Timestamp'], axis=1, inplace=True)
df.head()
# Further exploratory data analysis with respect to time



sns.countplot(x='Day of Week', hue='Clicked on Ad', data=df)

plt.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0)

plt.show()
sns.countplot(x='Month', hue='Clicked on Ad', data=df)

plt.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0)

plt.show()
plt.figure(figsize=(16,4))

sns.countplot(x='Hour', hue='Clicked on Ad', data=df)

plt.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0)

plt.show()
# Converting countries into dummy variables and dropping the first column. One Hot Encoding is also viable.

country = pd.get_dummies(df['Country'], drop_first=True)

country.shape
X = pd.concat([df, country], axis=1)

X.drop(['Country', 'Clicked on Ad'], axis=1, inplace=True)

y = df['Clicked on Ad']
# Generating training and testing datasets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print(X_train.shape)

print(X_test.shape)
# Importing model from sci-kit learn and fitting model with training data

from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression(solver='liblinear')

logmodel.fit(X_train, y_train)
# Predicitions based on model

y_pred = logmodel.predict(X_test)
# Model performance evaluation

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, y_pred))

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))