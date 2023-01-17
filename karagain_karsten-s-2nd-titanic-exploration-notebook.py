# Data Processing
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Visualizations
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning Classifiers
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import metrics

# Package to Access Dataset
import os
# Import training Dataset
df = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
full = pd.concat([df, test])
# Look at the relative distribution for the Fare
df.Fare.hist(bins=70)
# Get a statistical view of how the results are distributed. 
df.Fare.describe()
plt.figure(figsize=(6,6))
fare = 5
df.Pclass[df.Fare<=fare].hist()
plt.title('Price Less Than ' + str(fare))
plt.ylabel('Tickets Purchased')
sns.set_style(style='whitegrid')
for x in [5, 10, 50, 100, 500, 1000]:
    sns.factorplot("Pclass", "Survived", hue="Sex", data=df[df.Fare<=x], kind="bar", ci=None).fig.suptitle('Fare less than ' + str(x))
# sns.factorplot("Pclass", "Survived", hue="Sex", data=df[df.Fare<=50], kind="bar", ci=None).fig.suptitle('Fare less than 50')

# Changing the x variable allows you to view the tickets purchased below that price, and the class survivability probability. 
x = 50

df.Pclass[df.Fare<=x].hist()
plt.title('Price Less Than ' + str(x))
plt.ylabel('Tickets Purchased')

sns.factorplot("Pclass", "Survived", hue="Sex", data=df[df.Fare<=x], kind="bar", ci=None).fig.suptitle('Fare less than ' + str(x))

# Change the strings into numerals so the data can be recognized by the classifiers
df.Sex = df.Sex.str.replace('female', '0')
df.Sex = df.Sex.str.replace('male', '1')
df.Sex = df.Sex.astype('int')
# Separate the two testing cases
train = df[['Pclass', 'Sex']]

# Train and Run the GaussianNB classifier
features_train, features_test, labels_train, labels_test = train_test_split(train, df.Survived, test_size = 0.3, random_state = 42)
clf = GaussianNB()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

accuracy_score(labels_test, pred)
clf = LogisticRegression()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

accuracy_score(labels_test, pred)
# We recall that from the training data, 577 are men, 314 are female. 
df.Sex.value_counts()
# Visualizations; These graphs show that a large portion of women, and a little portion of men survived. 
plt.figure(figsize=(13,10))
plt.subplot(231)
plt.title('Female Survivors')
plt.bar(['Survived', 'Deceased'],df.Survived[df.Sex == 0].value_counts(normalize=True), color = ['c', 'm'])

plt.subplot(232)
plt.title('Male Survivors')
plt.bar(['Deceased', 'Survived'],df.Survived[df.Sex == 1].value_counts(normalize=True), color = ['r', 'b'])
# Here I create a new column to compare to Survived
df.loc[df.Sex == 0, 'pred'] = 1
df.loc[df.Sex == 1, 'pred'] = 0

# Get an accuracy score for those survived and this heuristic test
accuracy_score(df.Survived, df.pred)