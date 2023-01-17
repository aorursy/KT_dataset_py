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
train = pd.read_csv('../input/airline-passenger-satisfaction/train.csv')

test = pd.read_csv('../input/airline-passenger-satisfaction/test.csv')
train.info()
test.info()
print('Shape of the training set', train.shape)

print('Shape of the testing set', test.shape)
def missing_percent(df):

    total = df.isnull().sum().sort_values(ascending = False)

    percent = round(df.isnull().sum().sort_values(ascending = False)/ len(df) * 100, 2)

    return pd.concat([total, percent], axis = 1, keys = ['Total', '%'])
missing_percent(train)
missing_percent(test)
train.dropna(subset = ['Arrival Delay in Minutes'], inplace = True)

test.dropna(subset = ['Arrival Delay in Minutes'], inplace = True)
test_satisfaction = test.satisfaction

test.drop(columns = 'satisfaction', inplace = True)
import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('tableau-colorblind10')
sns.kdeplot(train.loc[(train['Gender'] == 'Male'), 'Age'], shade = True, color = 'b', label = 'Male')

sns.kdeplot(train.loc[(train['Gender'] == 'Female'), 'Age'], shade = True, color = 'g', label = 'Female')

plt.title('Age Distribution by Gender')
a = sns.FacetGrid(train, height = 3, row = 'Gender', col = 'Class', hue = 'Class', margin_titles = True)

a.map(plt.hist, 'Age', rwidth = 0.8)

a.fig.suptitle('Type of Travel by Gender/ Age')

plt.subplots_adjust(top = 0.90)
b = sns.FacetGrid(train, height = 5, row = 'Gender', col = 'Type of Travel', hue = 'Type of Travel')

b.map(plt.hist, 'Age', rwidth = 0.8)

b.fig.suptitle('Type of Travel by Gender/ Age')

plt.subplots_adjust(top = 0.90)
x = train['Flight Distance']

sns.distplot(x, bins = 15)

plt.title('Flight Distance Distribution')
train.columns
fig, ax = plt.subplots(figsize = (20, 4), ncols = 5)

ax1 = train.hist('Inflight wifi service', ax = ax[0], rwidth = 0.8)

ax2 = train.hist('Inflight entertainment', ax = ax[1], rwidth = 0.8)

ax3 = train.hist('Food and drink', ax = ax[2], rwidth = 0.8)

ax4 = train.hist('Seat comfort', ax = ax[3], rwidth = 0.8)

ax5 = train.hist('Leg room service', ax = ax[4], rwidth = 0.8)
fig, ax = plt.subplots(figsize = (20, 4), ncols = 5)

ax1 = train.hist('Departure/Arrival time convenient', ax = ax[0], rwidth = 0.8)

ax2 = train.hist('Ease of Online booking', ax = ax[1], rwidth = 0.8)

ax3 = train.hist('Gate location', ax = ax[2], rwidth = 0.8)

ax4 = train.hist('Baggage handling', ax = ax[3], rwidth = 0.8)

ax5 = train.hist('Cleanliness', ax = ax[4], rwidth = 0.8)
train['Delay'] = train['Departure Delay in Minutes'] + train['Arrival Delay in Minutes']
train.drop(columns = ['Departure Delay in Minutes', 'Arrival Delay in Minutes'], inplace = True)
train.satisfaction.replace({'satisfied' : 1, 'neutral or dissatisfied' : 0}, inplace = True)
sns.scatterplot(x = 'Delay', y = 'satisfaction', hue = 'Type of Travel', data = train)
test['Delay'] = test['Departure Delay in Minutes'] + test['Arrival Delay in Minutes']

test.drop(columns = ['Departure Delay in Minutes', 'Arrival Delay in Minutes'], inplace = True)
train['Comfort'] = (train['Seat comfort'] + train['Leg room service']) / 2

test['Comfort'] = (test['Seat comfort'] + test['Leg room service']) / 2
def replaceAge(df):

    if(df['Age'] <= 15):

        return "A"

    elif(df['Age'] > 15) & (df['Age'] <= 30):

        return "B"

    elif(df['Age'] > 30) & (df['Age'] <= 45):

        return "C"

    elif(df['Age'] > 46) & (df['Age'] <= 60):

        return "D"

    elif(df['Age'] > 61) & (df['Age'] <= 75):

        return "E"

    else:

        return "F"
train['Age'] = train.apply(lambda train: replaceAge(train), axis = 1)
test['Age'] = test.apply(lambda test: replaceAge(test), axis = 1)
train.head()
test.head()
train.drop(columns = ['Unnamed: 0', 'id'], inplace = True)

test.drop(columns = ['Unnamed: 0', 'id'], inplace = True)
def replaceDist(df):

    if(df['Flight Distance'] <= 500):

        return "A"

    elif(df['Flight Distance'] > 500) & (df['Flight Distance'] <= 1000):

        return "B"

    elif(df['Flight Distance'] > 1000) & (df['Flight Distance'] <= 1500):

        return "C"

    elif(df['Flight Distance'] > 1500) & (df['Flight Distance'] <= 2000):

        return "D"

    elif(df['Flight Distance'] > 2000) & (df['Flight Distance'] <= 2500):

        return "E"

    elif(df['Flight Distance'] > 2500) & (df['Flight Distance'] <= 3000):

        return "F"

    elif(df['Flight Distance'] > 3000) & (df['Flight Distance'] <= 3500):

        return "G"

    elif(df['Flight Distance'] > 3500) & (df['Flight Distance'] <= 4000):

        return "H"

    else:

        return "I"
train['Flight Distance'] = train.apply(lambda train:replaceDist(train), axis = 1)
train.head()
test['Flight Distance'] = test.apply(lambda test:replaceDist(test), axis = 1)
test.head()
train_num_cols = train.select_dtypes(['int64', 'float64']).columns.tolist()

test_num_cols = test.select_dtypes(['int64', 'float64']).columns.tolist()
train_obj_cols = train.select_dtypes('object').columns.tolist()

test_obj_cols = test.select_dtypes('object').columns.tolist()
for i in train_obj_cols:

    train[i] = train[i].astype('category')
for i in test_obj_cols:

    test[i] = test[i].astype('category')
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
train[train_obj_cols] = train[train_obj_cols].apply(lambda col : le.fit_transform(col))
train.head()
test[test_obj_cols] = test[test_obj_cols].apply(lambda col : le.fit_transform(col))
test.head()
fig, ax = plt.subplots(figsize = (16,12))

sns.heatmap(abs(train.corr()) * 100, annot = True)
train_df = train.drop(columns = ['Gender', 'Customer Type', 'Age', 'Departure/Arrival time convenient', 'Ease of Online booking',

                                'Gate location', 'Food and drink', 'Delay'])

test_df = test.drop(columns = ['Gender', 'Customer Type', 'Age', 'Departure/Arrival time convenient', 'Ease of Online booking',

                                'Gate location', 'Food and drink', 'Delay'])
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
LR = LogisticRegression()
X = train_df.drop('satisfaction', axis = 1)

Y = train_df['satisfaction']
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size = 0.2, random_state = 0)
LR.fit(train_x, train_y)
pred = LR.predict(test_x)
accuracy_score(pred, test_y)
from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier
models = []

models.append(('Decision Tree', DecisionTreeClassifier()))

models.append(('Random Forest', RandomForestClassifier()))

models.append(('Naive Bayes', GaussianNB()))

#models.append(('SVM', SVC()))

models.append(('KNN', KNeighborsClassifier()))
from sklearn import model_selection

for name, model in models:

    kfold = model_selection.KFold(n_splits = 5, random_state = 0, shuffle = True)

    cv_score = model_selection.cross_val_score(model, train_x, train_y, cv = kfold, scoring = 'accuracy')

    print(name, ': ', cv_score.mean())
RF = RandomForestClassifier()

RF.fit(train_x, train_y)

pred = RF.predict(test_x)

print('Accuracy on Training set: ', accuracy_score(pred, test_y))
from sklearn import metrics

print(metrics.classification_report(pred, test_y))
print(metrics.confusion_matrix(pred, test_y))
pred_test = RF.predict(test_df)
test_satisfaction.replace({'satisfied' : 1, 'neutral or dissatisfied' : 0}, inplace = True)
print('Accuracy on Testing set: ', accuracy_score(pred_test, test_satisfaction))
print(metrics.classification_report(pred_test, test_satisfaction))
print(metrics.confusion_matrix(pred_test, test_satisfaction))