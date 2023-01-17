# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Visualization
import seaborn as sns 
import re
import matplotlib.pyplot as plt
# Datetime
from datetime import datetime
# Sklearn import
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import preprocessing
# Text processing
from textblob import TextBlob
import string 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
train_data = pd.read_csv("../input/train.csv")
# Overview
train_data.head()
train_data.info()
# Find if any entries are null
for i in train_data.columns:
    print(i, train_data[i].isnull().sum().sum())
# Fill in missing data by empty string
train_data['name'].fillna(" ")
train_data['desc'].fillna(" ")
# Convert UNIX time format to standard time format
date_column = ['deadline', 'state_changed_at', 'created_at', 'launched_at']
for i in date_column:
    train_data[i]=train_data[i].apply(lambda x: datetime.fromtimestamp(int(x)).strftime("%Y-%m-%d %H:%M:%S"))
# Distribution of funded projects
sns.countplot(x='final_status',data=train_data)
plt.show()
# Distribution of goals
sns.distplot(train_data['goal'], bins=5)
plt.show()
train_data['goal'].describe()
#Remove some of the outliers and replot the histograms
P = np.percentile(train_data['goal'], [0, 95])
new_goal = train_data[(train_data['goal'] > P[0]) & (train_data['goal'] < P[1])]
sns.distplot(new_goal['goal'], bins=5)
plt.show()
# Log-transform goal without excluding outliers 
sns.distplot(np.log(train_data['goal']), bins=5)
plt.show()
g = sns.FacetGrid(new_goal, col='final_status')
g.map(plt.hist, 'goal', bins = 40)
plt.show()
g = sns.FacetGrid(new_goal, col="final_status",  row="country")
g = g.map(plt.hist, "goal", bins = 40)
plt.show()
#non_us = new_goal[new_goal['country'] != 'US']
train_data['log_goal'] = np.log(train_data['goal'])
g = sns.FacetGrid(train_data, col="final_status",  row="country")
g = g.map(plt.hist, "log_goal", bins = 40)
# Explore the effect of disable_communication
figure, axes = plt.subplots(1, 2, sharey=True)
sns.countplot(x='disable_communication',data=train_data, hue='final_status', ax = axes[0])
sns.countplot(x='final_status', data= train_data, ax = axes[1])
plt.show()
train_data['disable_communication'].describe()
figure, axes = plt.subplots(2)
sns.countplot(x='country',data=train_data, hue='final_status', ax = axes[0])
sns.countplot(x='currency',data=train_data, hue='final_status', ax = axes[1])
plt.show()
figure, axes = plt.subplots(2)
sns.countplot(x='country',data=train_data, ax = axes[0])
sns.countplot(x='currency',data=train_data, ax = axes[1])
plt.show()
# Understand the distribution of backers using box-plot
ax = sns.boxplot(x=train_data["backers_count"])

#Remove some of the outliers and replot the histograms
P_backer = np.percentile(train_data['backers_count'], [0, 95])
new_backers = train_data[(train_data['backers_count'] > P_backer[0]) & (train_data['backers_count'] < P_backer[1])]
ax = sns.boxplot(x=new_backers["backers_count"])
new_backers.shape
# Explore the effect of disable_communication
# figure, axes = plt.subplots(1, 2, sharey=True)
sns.countplot(x='backers_count',data=new_backers, hue='final_status')
plt.xticks([],[])
# sns.countplot(x='final_status', data= train_data, ax = axes[1])
plt.show()
g = sns.FacetGrid(new_backers, col="final_status",  row="country")
g = g.map(plt.hist, "backers_count", bins = 40)
plt.scatter(new_backers['backers_count'], np.log(new_backers['goal']), alpha = 0.3)
plt.xlabel('backers count')
plt.ylabel('log goal')
plt.show()
plt.scatter(new_backers[new_backers['final_status'] == 1]['backers_count'], np.log(new_backers[new_backers['final_status'] == 1]['goal']), alpha = 0.3)
plt.xlabel('backers count')
plt.ylabel('log goal')
plt.title("Funded project")
plt.show()
plt.scatter(new_backers[new_backers['final_status'] == 0]['backers_count'], np.log(new_backers[new_backers['final_status'] == 0]['goal']), alpha = 0.3)
plt.xlabel('backers count')
plt.ylabel('log goal')
plt.title("Not funded project")
plt.show()
# with respect to launched time 
def countQuarter(dt):
    month = int(dt[5:7])
    if month <= 3: return '01'
    elif month <= 6:return '02'
    elif month <= 9: return '03'
    else: return '04'

train_data['launched_month'] = train_data['launched_at'].apply(lambda dt: dt[5:7])
train_data['launched_year'] = train_data['launched_at'].apply(lambda dt: dt[0:4])
train_data['launched_quarter'] = train_data['launched_at'].apply(lambda dt: countQuarter(dt))
figure, axes = plt.subplots(3)
sns.countplot(x='launched_month',data=train_data, hue='final_status', ax = axes[0])
sns.countplot(x='launched_year',data=train_data, hue='final_status', ax = axes[1])
sns.countplot(x='launched_quarter',data=train_data, hue='final_status', ax = axes[2])
plt.tight_layout()
plt.show()
def measureDuration(dt): # Duration in hours
    launch = datetime.strptime(dt[0], "%Y-%m-%d %H:%M:%S")
    deadline = datetime.strptime(dt[1], "%Y-%m-%d %H:%M:%S")
    difference = deadline-launch
    hr_difference = int (difference.total_seconds() / 3600)
    return hr_difference

train_data['duration'] = train_data[['launched_at', 'deadline']].apply(lambda dt: measureDuration(dt), axis=1)
sns.distplot(train_data['duration'], bins=5)
plt.show()
def measureDurationByWeek(dt):
    # count by hr / week 
    week = 168 
    return int (dt / 168)

train_data['duration_weeks'] = train_data['duration'].apply(lambda dt: measureDurationByWeek(dt))
sns.countplot(x='duration_weeks', data=train_data, hue='final_status')
plt.show()
train_data.head()
train_data.info()
def getFeatures(x_features, y_feature): 
    X = train_data[x_features]
    y = train_data[y_feature]
    return X, y

def splitData(X, y, size): 
    onehot_X = pd.get_dummies(X)
    X_train, X_test, y_train, y_test = train_test_split(onehot_X, y, test_size=size, random_state = 42)
    return X_train, X_test, y_train, y_test
    
def makeLogisticRegression(X_train, y_train): 
    lr = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
    clf = lr.fit(X_train, y_train)
    return lr, clf 

def accuracy(clf, X_train, X_test, y_train, y_test):
    return clf.score(X_train, y_train), clf.score(X_test, y_test)
x_features = ['log_goal','country','currency', 'backers_count', 'launched_year','duration_weeks']
y_feature = 'final_status'
X, y = getFeatures(x_features, y_feature)
X_train, X_test, y_train, y_test = splitData(X, y, 0.2)
lr, clf = makeLogisticRegression(X_train, y_train)
train_score, test_score = accuracy(clf, X_train, X_test, y_train, y_test)
train_score
test_score
x_features = ['log_goal','country','currency', 'backers_count', 'launched_month', 'launched_year','duration_weeks']
y_feature = 'final_status'
X, y = getFeatures(x_features, y_feature)
X_train, X_test, y_train, y_test = splitData(X, y, 0.2)
lr, clf = makeLogisticRegression(X_train, y_train)
train_score, test_score = accuracy(clf, X_train, X_test, y_train, y_test)
train_score
test_score
lr.coef_
# Length of Project Name 
train_data['name_length'] = train_data['name'].apply(lambda name: len(str(name)))
sns.countplot(x='name_length', data=train_data, hue='final_status')
plt.show()
P = np.percentile(train_data['name_length'], [5, 95])
parsed_name = train_data[(train_data['name_length'] > P[0]) & (train_data['name_length'] < P[1])]

sns.distplot(parsed_name['name_length'], bins=10)
plt.show()
# Project name in alphabetical order
def parseName(name):
    if str(name)[0] not in string.ascii_lowercase + string.ascii_uppercase: 
        return '*'
    else:
        return str(name)[0].lower()

train_data['alpha_order'] = train_data['name'].apply(lambda name: parseName(name))
sns.countplot(x='alpha_order', data=train_data, hue='final_status')
plt.show()
x_features = ['log_goal','country', 'currency', 'backers_count', 'launched_year', 'launched_month', 'duration_weeks', 'name_length', 'alpha_order']
y_feature = 'final_status'
X, y = getFeatures(x_features, y_feature)
X_train, X_test, y_train, y_test = splitData(X, y, 0.2)
lr, clf = makeLogisticRegression(X_train, y_train)
train_score, test_score = accuracy(clf, X_train, X_test, y_train, y_test)
train_score
test_score
# Keyword Search 

buzzwords = ['app', 'platform', 'technology', 'service', 'solution', 'data', 
            'manage', 'market', 'help', 'mobile', 'users', 'system', 'software', 
           'customer', 'application', 'online', 'web', 'create', 'health', 
           'provider', 'network', 'cloud', 'social', 'device', 'access']

def countBuzzwords(desc):
    lowerCase = str(desc).lower() 
    count = 0
    for bw in buzzwords: 
        count += lowerCase.count(bw)
    return count 
    
train_data['buzzword_count'] = train_data['desc'].apply(lambda d: countBuzzwords(d))
sns.countplot(x='buzzword_count', data=train_data, hue='final_status')
plt.show()
x_features = ['log_goal','country', 'currency', 'backers_count', 'launched_year', 'launched_month', 'duration_weeks', 'name_length', 'alpha_order', 'buzzword_count']
y_feature = 'final_status'
X, y = getFeatures(x_features, y_feature)
X_train, X_test, y_train, y_test = splitData(X, y, 0.2)
lr, clf = makeLogisticRegression(X_train, y_train)
train_score, test_score = accuracy(clf, X_train, X_test, y_train, y_test)
train_score
test_score
x_features = ['log_goal','country', 'currency', 'backers_count', 'launched_year', 'launched_month', 'duration_weeks', 'buzzword_count']
y_feature = 'final_status'
X, y = getFeatures(x_features, y_feature)
X_train, X_test, y_train, y_test = splitData(X, y, 0.2)
lr, clf = makeLogisticRegression(X_train, y_train)
train_score, test_score = accuracy(clf, X_train, X_test, y_train, y_test)
train_score
test_score
