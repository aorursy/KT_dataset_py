import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import os



# Libraries for model building and evaluation

from sklearn import metrics

from sklearn import preprocessing

from sklearn.preprocessing import PowerTransformer

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import auc

import imblearn

from imblearn.over_sampling import SMOTE
# Read data from csv file

df = pd.read_csv('../input/creditcardfraud/creditcard.csv')

df.head()
# Observed the different feature types present in the data

df.info()
classes=df['Class'].value_counts()

normal_share=classes[0]/df['Class'].count()*100

fraud_share=classes[1]/df['Class'].count()*100
# Created a bar plot for the number and percentage of fraudulent vs non-fraudulent transactions

labels = ['Non-fradulent', 'Fradulent']

perc = [normal_share, fraud_share]

plt.figure(figsize=(8,6))

plt.title('Non-fradulent vs Fradulent Transactions')

sns.barplot(x=labels, y=perc)

for i in range(len(classes)):

  plt.text(x=i, y=perc[i], s='{0} ({1:.2f}%)'.format(classes[i],perc[i]), ha='center', va='bottom', size='large')
# Created a scatter plot to observe the distribution of classes with time

plt.figure(figsize=(12,8))

plt.title('Class vs Time')

sns.scatterplot(x='Time', y='Class', data=df)

plt.yticks(classes.index)

plt.ylim(classes.index[0]-0.5, classes.index[1]+0.5);
# Created a scatter plot to observe the distribution of classes with Amount

plt.figure(figsize=(12,8))

plt.title('Class vs Amount')

sns.scatterplot(x='Amount', y='Class', data=df)

plt.yticks(classes.index)

plt.ylim(classes.index[0]-0.5, classes.index[1]+0.5);
# Created a scatter plot to observe the distribution of classes with Amount

plt.figure(figsize=(12,8))

plt.title('Class vs Amount')

sns.scatterplot(x="Time", y='Amount',hue='Class', data=df)

plt.yticks(classes.index)

plt.ylim(classes.index[0]-0.5, classes.index[1]+0.5);
plt.rcParams["figure.figsize"] = (12,10)

df["hour_of_day"] = (df["Time"]%(3600*24))//3600



plt.subplot(2,1,1)

# plt.title('Class vs Amount')

sns.scatterplot(x="hour_of_day", y='Amount',hue='Class', data=df)

plt.subplot(2,1,2)

plt.yscale("log")

sns.boxplot(x="hour_of_day",y="Amount", hue="Class",data =df)
plt.figure(figsize=(20,50))

for i in range (1,28):

    plt.subplot(7,4,i)

    sns.scatterplot(x="V"+str(i), y ="V"+str(i+1),hue = "Class",data =df)

#     plt.tight_layout()
# Drop unnecessary columns

df = df.drop('Time', axis=1)

df.shape
y = df.pop('Class') #class variable
# Split the data into train and test sets

from sklearn import model_selection



X_train, X_test, y_train, y_test = model_selection.train_test_split(df, y, train_size=0.7, test_size=0.3,

                                                                    stratify=y, random_state=42)
# Check number of fradulent transactions

print(np.sum(y))

print(np.sum(y_train))

print(np.sum(y_test))
# Plot the histograms of variables from the dataset to see the skewness

cols = X_train.columns

c = 5

r = np.ceil(len(cols)/c)

plt.figure(figsize=(4*c, 4*r))

for i in range(len(cols)):

  plt.subplot(r, c, i+1)

  sns.distplot(X_train[cols[i]])
# Apply: preprocessing.PowerTransformer(copy=False) to fit & transform the train & test data

from sklearn import preprocessing



pt = preprocessing.PowerTransformer(copy=False)

cols_skewed = ['V1', 'Amount']

for col in cols_skewed:

  pt.fit_transform(X_train[col].values.reshape(-1, 1))

  pt.transform(X_test[col].values.reshape(-1, 1))
# Plot the histograms of transformed variables from the dataset to see the result

c = 5

r = np.ceil(len(cols_skewed)/c)

plt.figure(figsize=(4*c, 4*r))

for i in range(len(cols_skewed)):

  plt.subplot(r, c, i+1)

  sns.distplot(X_train[cols_skewed[i]])
from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression(random_state=42)

logreg.fit(X_train, y_train)
print("Train accuracy",logreg.score(X_train, y_train))

print("Test accuracy",logreg.score(X_test, y_test))
from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(n_jobs=-1,criterion = 'entropy', max_depth = 5, min_samples_leaf=5, random_state=42)

rf.fit(X_train, y_train)
print("Train accuracy",rf.score(X_train, y_train))

print("Test accuracy",rf.score(X_test, y_test))
from xgboost import XGBClassifier



xgb = XGBClassifier(n_jobs=-1, random_state=42, n_estimators=120, max_depth = 5, min_samples_leaf=5)

xgb.fit(X_train, y_train)
print("Train accuracy",xgb.score(X_train, y_train))

print("Test accuracy",xgb.score(X_test, y_test))