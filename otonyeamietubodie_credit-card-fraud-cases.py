import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import plotly_express as px

import matplotlib.image as mpimg

from tabulate import tabulate

import missingno as msno 

from IPython.display import display_html

from PIL import Image

import gc

import cv2

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
data.head(5)
data.info()
data.describe(include='all')
data.isnull().sum().sort_values(ascending=False)
data.columns
f, ax = plt.subplots(figsize=(12, 5))

sns.countplot('Class', data=data)
#The classes in the dataset are very skewed and very imbalanced 

# As you can see most of the transactions are non fraud if we should go ahead 

# and use this imbalanced dataset and make our predictions our results might be wrong.
#Lets check the distribution of amount and some other features
f, ax = plt.subplots(figsize=(12, 5))

sns.distplot(data['Amount'], color='r')
f, ax = plt.subplots(figsize=(12,5))

sns.distplot(data.Time, color='g')
# From the dataset we can observe all the features in the dataset are scaled 

#except for Amount and Time.



#Since the dataset is also imbalanced we can  divide the dataset into subsamples 

# so we can have an equal amount of fraud and non fraud cases so as to help the model 

# understand the data more
from sklearn.preprocessing import StandardScaler, RobustScaler
robust_scaler = RobustScaler()
data['scaled_amount'] = robust_scaler.fit_transform(data['Amount'].values.reshape(-1, 1))

data['Time'] = robust_scaler.fit_transform(data['Time'].values.reshape(-1, 1))
data.head(5)
data.drop(['Time','Amount'], axis=1, inplace=True)
from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedShuffleSplit
df = data.copy()
data = data.sample(frac = 1)
fraud_data = data.loc[data['Class'] == 1]

non_fraud_data = data.loc[data['Class'] == 0][:492]
normally_distributed = pd.concat([fraud_data, non_fraud_data])
new_data = normally_distributed.sample(frac=1, random_state=1)
new_data.head(5)
colors = ["#0101DF", "#DF0101"]

f, ax = plt.subplots(figsize=(12, 5))

sns.countplot('Class', data=new_data, palette=colors)
#correlation matrix helps us understand our data. It helps us to understand the 

#correlation between two vaariables. If there are features that heavily influence

#whether a specific transaction is fraudulent or not
#Below is a heatmap of the correlation of the normal data:

correlation_matrix = data.corr()

fig = plt.figure(figsize=(20,8))

sns.heatmap(correlation_matrix, vmax=0.8, square=True)
#Below is a heatmap of the correlation of the normal data:

correlation_matrix = new_data.corr()

fig = plt.figure(figsize=(20,8))

sns.heatmap(correlation_matrix, vmax=0.8, square=True)
correlation_matrix['Class'].sort_values(ascending=False)
#My main aim is to remove extreme outliers from features that have high correlation with our classes

#we can remove these outliers by using interquartile range method
v10 = df['V10'].loc[df['Class'] == 1].values

v12 = df['V12'].loc[df['Class'] == 1].values

v14 = df['V14'].loc[df['Class'] == 1].values
f, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(20, 6))



sns.distplot(v10, ax=ax1, color='g')

ax1.set_title('V10 Distribution Fraud Transactions)', fontsize=14)





sns.distplot(v12, ax=ax2, color='r')

ax2.set_title('V12 Distribution Fraud Transactions)', fontsize=14)



sns.distplot(v14, ax=ax3, color='b')

ax3.set_title('V14 Distribution Fraud Transactions)', fontsize=14)
# # -----> V14 Removing Outliers (Highest Negative Correlated with Labels)

v14_fraud = df['V14'].loc[df['Class'] == 1].values

q25, q75 = np.percentile(v14_fraud, 25), np.percentile(v14_fraud, 75)

print('Quartile 25: {} | Quartile 75: {}'.format(q25, q75))

v14_iqr = q75 - q25

print('iqr: {}'.format(v14_iqr))



v14_cut_off = v14_iqr * 1.5

v14_lower, v14_upper = q25 - v14_cut_off, q75 + v14_cut_off

print('Cut Off: {}'.format(v14_cut_off))

print('V14 Lower: {}'.format(v14_lower))

print('V14 Upper: {}'.format(v14_upper))



outliers = [x for x in v14_fraud if x < v14_lower or x > v14_upper]

print('Feature V14 Outliers for Fraud Cases: {}'.format(len(outliers)))

print('V10 outliers:{}'.format(outliers))



new_data = new_data.drop(new_data[(new_data['V14'] > v14_upper) | (new_data['V14'] < v14_lower)].index)

print('----' * 44)



# -----> V12 removing outliers from fraud transactions

v12_fraud = new_data['V12'].loc[new_data['Class'] == 1].values

q25, q75 = np.percentile(v12_fraud, 25), np.percentile(v12_fraud, 75)

v12_iqr = q75 - q25



v12_cut_off = v12_iqr * 1.5

v12_lower, v12_upper = q25 - v12_cut_off, q75 + v12_cut_off

print('V12 Lower: {}'.format(v12_lower))

print('V12 Upper: {}'.format(v12_upper))

outliers = [x for x in v12_fraud if x < v12_lower or x > v12_upper]

print('V12 outliers: {}'.format(outliers))

print('Feature V12 Outliers for Fraud Cases: {}'.format(len(outliers)))

new_data = new_data.drop(new_data[(new_data['V12'] > v12_upper) | (new_data['V12'] < v12_lower)].index)

print('Number of Instances after outliers removal: {}'.format(len(new_data)))

print('----' * 44)





# Removing outliers V10 Feature

v10_fraud = new_data['V10'].loc[new_data['Class'] == 1].values

q25, q75 = np.percentile(v10_fraud, 25), np.percentile(v10_fraud, 75)

v10_iqr = q75 - q25



v10_cut_off = v10_iqr * 1.5

v10_lower, v10_upper = q25 - v10_cut_off, q75 + v10_cut_off

print('V10 Lower: {}'.format(v10_lower))

print('V10 Upper: {}'.format(v10_upper))

outliers = [x for x in v10_fraud if x < v10_lower or x > v10_upper]

print('V10 outliers: {}'.format(outliers))

print('Feature V10 Outliers for Fraud Cases: {}'.format(len(outliers)))

new_data = new_data.drop(new_data[(new_data['V10'] > v10_upper) | (new_data['V10'] < v10_lower)].index)

print('Number of Instances after outliers removal: {}'.format(len(new_data)))
y = new_data['Class']

X = new_data.drop(['Class'], axis=1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.values

X_test = X_test.values

y_train = y_train.values

y_test = y_test.values
from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score
classifier = LogisticRegression()

classifier.fit(X_train, y_train)
training_score = cross_val_score(classifier, X_train, y_train, cv=10)
training_score
training_score.mean()
# Use GridSearchCV to find the best parameters.

from sklearn.model_selection import GridSearchCV





# Logistic Regression 

log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}







grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params)

grid_log_reg.fit(X_train, y_train)

# We automatically get the logistic regression with the best parameters.

log_reg = grid_log_reg.best_estimator_

log_reg
log_reg_score = cross_val_score(log_reg, X_train, y_train, cv=10)
log_reg_score
y_pred = log_reg.predict(X_test)
y_pred
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred, y_test)
cm
from sklearn.metrics import classification_report
print('Logistic Regression:')

print(classification_report(y_test, y_pred))