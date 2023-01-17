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
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import re



import warnings

warnings.filterwarnings('ignore')
from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.metrics import classification_report, confusion_matrix



from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from xgboost import XGBClassifier
train_df = pd.read_csv('../input/titanic/train.csv')

print(train_df.shape)

train_df.head()
test_df = pd.read_csv('../input/titanic/test.csv')

print(test_df.shape)

test_df.head()
train_df.info()
train_df.describe()
sns.heatmap(train_df.corr(), annot = True)
sns.set_style('darkgrid')

fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(20, 6))

sns.barplot(x = 'SibSp', y = 'Survived', data = train_df, ax = ax1, palette='coolwarm')

sns.barplot(x = 'Parch', y = 'Survived', data = train_df, ax = ax2, palette = 'magma')

sns.barplot(x = 'Embarked', y = 'Survived', data = train_df, ax = ax3)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

females = train_df[train_df['Sex'] == 'female']

males = train_df[train_df['Sex'] == 'male']



ax = sns.distplot(females[females['Survived'] == 1].Age, bins=30, label='Survived', ax=axes[0])

ax = sns.distplot(females[females['Survived'] == 0].Age, bins=30, label='Not Survived', ax=axes[0])

ax.legend()

ax.set_title('Female')

ax = sns.distplot(males[males['Survived'] == 1].Age, bins=30, label='Survived', ax=axes[1])

ax = sns.distplot(males[males['Survived'] == 0].Age, bins=30, label='Not Survived', ax=axes[1], )

ax.legend()

ax.set_title('Male')
df = [train_df, test_df]
for data in df:

    data['Title'] = data['Name'].str.extract(r', (\w+)\.', expand=False)

pd.crosstab(train_df['Title'], train_df['Sex']).transpose()
for data in df:

    data['Title'] = data['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    data['Title'] = data['Title'].replace('Mlle', 'Miss')

    data['Title'] = data['Title'].replace('Ms', 'Miss')

    data['Title'] = data['Title'].replace('Mme', 'Mrs')

    

train_df[['Title', 'Survived']].groupby(['Title']).mean()



labels = {'Mr':1, 'Mrs':2, 'Master':3, 'Miss':4, 'Rare':5}

test_df.replace({'Title':labels}, inplace = True)

train_df.replace({'Title':labels}, inplace = True)

train_df['Title'] = train_df['Title'].fillna(0)

train_df['Title'] = train_df['Title'].astype(int)                     # this is performed beacuse it was giving float values of title
pd.DataFrame({'Train':train_df.isnull().sum(), 'Test':test_df.isnull().sum()}).transpose()
print('Missing Values in Age column: ',177/len(train_df['Age'])*100)

print('Missing Values in Cabin column: ',687/len(train_df['Cabin'])*100)

print('Missing Values in Embarked column: ',2/len(train_df['Embarked'])*100)
fig, (ax1, ax2) = plt.subplots(ncols = 2, figsize = (15,5))

sns.heatmap(train_df.isnull(), cmap = 'coolwarm', ax = ax1)

sns.heatmap(test_df.isnull(), cmap = 'mako_r', ax = ax2)
train_df["Age"] = train_df["Age"].fillna(-0.5)

test_df["Age"] = test_df["Age"].fillna(-0.5)

bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]

labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

train_df['AgeGroup'] = pd.cut(train_df["Age"], bins, labels = labels)

test_df['AgeGroup'] = pd.cut(test_df["Age"], bins, labels = labels)

mr_age = train_df[train_df["Title"] == 1]["AgeGroup"].mode() #Young Adult

miss_age = train_df[train_df["Title"] == 2]["AgeGroup"].mode() #Student

mrs_age = train_df[train_df["Title"] == 3]["AgeGroup"].mode() #Adult

master_age = train_df[train_df["Title"] == 4]["AgeGroup"].mode() #Baby

rare_age = train_df[train_df["Title"] == 5]["AgeGroup"].mode() #Adult



age_title_mapping = {1: "Young Adult", 2: "Student", 3: "Adult", 4: "Baby", 5: "Adult"}



for x in range(len(train_df["AgeGroup"])):

    if train_df["AgeGroup"][x] == "Unknown":

        train_df["AgeGroup"][x] = age_title_mapping[train_df["Title"][x]]

        

for x in range(len(test_df["AgeGroup"])):

    if test_df["AgeGroup"][x] == "Unknown":

        test_df["AgeGroup"][x] = age_title_mapping[test_df["Title"][x]]
df_m = train_df[train_df['Survived'] == 0]

df_f = train_df[train_df['Survived'] == 1]

df_m = df_m['AgeGroup'].value_counts()

df_f = df_f['AgeGroup'].value_counts()



trace1 = go.Bar(x = df_m.index[::-1], y = df_m.values[::-1], name = 'Not Survived', marker = dict(color = 'dodgerblue'))

trace2 = go.Bar(x = df_f.index[::-1], y = df_f.values[::-1], name = 'Survived', marker = dict(color = 'deeppink'))

data = [trace1, trace2]

layout = go.Layout(height = 400, width = 500, title='Age Distribution')

fig = go.Figure(data = data, layout= layout)

py.iplot(fig)
age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7}

train_df['AgeGroup'] = train_df['AgeGroup'].map(age_mapping).astype(int)

test_df['AgeGroup'] = test_df['AgeGroup'].map(age_mapping).astype(int)
df_m = train_df[train_df['Sex'] == 'male']

df_f = train_df[train_df['Sex'] == 'female']

df_m = df_m['Embarked'].value_counts()

df_f = df_f['Embarked'].value_counts()



trace1 = go.Bar(x = df_m.index[::-1], y = df_m.values[::-1], name = 'Male', marker = dict(color = 'indigo'))

trace2 = go.Bar(x = df_f.index[::-1], y = df_f.values[::-1], name = 'Female', marker = dict(color = 'green'))

data = [trace1, trace2]

layout = go.Layout(height = 400, width = 500, title='Embarked Distribution with Sex')

fig = go.Figure(data = data, layout= layout)

py.iplot(fig)
df_m = train_df[train_df['Survived'] == 0]

df_f = train_df[train_df['Survived'] == 1]

df_m = df_m['Embarked'].value_counts()

df_f = df_f['Embarked'].value_counts()



trace1 = go.Bar(x = df_m.index[::-1], y = df_m.values[::-1], name = 'Male', marker = dict(color = 'burlywood'))

trace2 = go.Bar(x = df_f.index[::-1], y = df_f.values[::-1], name = 'Female', marker = dict(color = 'cadetblue'))

data = [trace1, trace2]

layout = go.Layout(height = 400, width = 500, title='Embarked Distribution with Survived')

fig = go.Figure(data = data, layout= layout)

py.iplot(fig)
train_df['Embarked'].fillna('S', inplace = True)



label = {'S':1, 'C':2, 'Q':3}

train_df.replace({'Embarked':label}, inplace = True)

test_df.replace({'Embarked':label}, inplace = True)
train_df['Cabin'] = train_df['Cabin'].fillna('X')

test_df['Cabin']=test_df['Cabin'].fillna('X')
for data in df:

    data['Cabin'] = data['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())

    

category = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7, 'X':8, 'T':9}

for data in df:

    data['Cabin'] = data['Cabin'].map(category)
df_m = train_df[train_df['Survived'] == 0]

df_f = train_df[train_df['Survived'] == 1]

df_m = df_m['Cabin'].value_counts()

df_f = df_f['Cabin'].value_counts()



trace1 = go.Bar(x = df_m.index[::-1], y = df_m.values[::-1], name = 'Not Survived', marker = dict(color = 'chartreuse'))

trace2 = go.Bar(x = df_f.index[::-1], y = df_f.values[::-1], name = 'Survived', marker = dict(color = 'darkred'))

data = [trace1, trace2]

layout = go.Layout(height = 400, width = 500, title='Cabin Distribution')

fig = go.Figure(data = data, layout= layout)

py.iplot(fig)
test_df['Fare'].fillna(test_df['Fare'].median(), inplace = True)



train_df['Fare'] = pd.qcut(train_df['Fare'], 4, labels = [1, 2, 3, 4])

test_df['Fare'] = pd.qcut(test_df['Fare'], 4, labels = [1, 2, 3, 4])
df_m = train_df[train_df['Survived'] == 0]

df_f = train_df[train_df['Survived'] == 1]

df_m = df_m['Fare'].value_counts()

df_f = df_f['Fare'].value_counts()



trace1 = go.Bar(x = df_m.index[::-1], y = df_m.values[::-1], name = 'Not Survived', marker = dict(color = 'coral'))

trace2 = go.Bar(x = df_f.index[::-1], y = df_f.values[::-1], name = 'Survived', marker = dict(color = 'teal'))

data = [trace1, trace2]

layout = go.Layout(height = 400, width = 500, title='Fare Distribution')

fig = go.Figure(data = data, layout= layout)

py.iplot(fig)
#if we check the data info then fare feature is a category not int, so to convert we are performing this: 

train_df['Fare'] = pd.to_numeric(train_df['Fare'])
df_m = train_df[train_df['Sex'] == 'male']

df_f = train_df[train_df['Sex'] == 'female']

df_m = df_m['Survived'].value_counts()

df_f = df_f['Survived'].value_counts()



trace1 = go.Bar(x = df_m.index[::-1], y = df_m.values[::-1], name = 'Male', marker = dict(color = 'lightseagreen'))

trace2 = go.Bar(x = df_f.index[::-1], y = df_f.values[::-1], name = 'Female', marker = dict(color = 'crimson'))

data = [trace1, trace2]

layout = go.Layout(height = 400, width = 500, title='Survival Distribution')

fig = go.Figure(data = data, layout= layout)

py.iplot(fig)
label = {'male':1, 'female':0}

train_df.replace({'Sex':label}, inplace = True)

test_df.replace({'Sex':label}, inplace = True)
df_m = train_df[train_df['Survived'] == 0]

df_f = train_df[train_df['Survived'] == 1]

df_m = df_m['Pclass'].value_counts()

df_f = df_f['Pclass'].value_counts()



trace1 = go.Bar(x = df_m.index[::-1], y = df_m.values[::-1], name = 'Not Survived', marker = dict(color = 'firebrick'))

trace2 = go.Bar(x = df_f.index[::-1], y = df_f.values[::-1], name = 'Survived', marker = dict(color = 'gold'))

data = [trace1, trace2]

layout = go.Layout(height = 400, width = 500, title='Pclass Distribution', )

fig = go.Figure(data = data, layout= layout)

py.iplot(fig)
for data in df:

    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1



train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for data in df:

    data['IsAlone'] = 0

    data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1



train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
df_m = train_df[train_df['Survived'] == 0]

df_f = train_df[train_df['Survived'] == 1]

df_m = df_m['IsAlone'].value_counts()

df_f = df_f['IsAlone'].value_counts()



trace1 = go.Bar(x = df_m.index[::-1], y = df_m.values[::-1], name = 'Not Survived', marker = dict(color = 'seagreen'))

trace2 = go.Bar(x = df_f.index[::-1], y = df_f.values[::-1], name = 'Survived', marker = dict(color = 'aqua'))

data = [trace1, trace2]

layout = go.Layout(height = 400, width = 500, title='IsAlone Distribution', )

fig = go.Figure(data = data, layout= layout)

py.iplot(fig)
train_df.head(1)
test_df.head(1)
train_df.drop(['PassengerId', 'Name', 'Ticket', 'Age', 'SibSp', 'Parch', 'FamilySize'], axis = 1, inplace = True)

test_df.drop(['Name', 'Ticket', 'Age', 'SibSp', 'Parch', 'FamilySize'], axis = 1, inplace = True)
train_df.head(3)
test_df.head(3)
X = train_df.drop('Survived', axis = 1)

y = train_df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 123)
from sklearn.metrics import accuracy_score
lr = LogisticRegression()

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)



print('Classification Report: \n', classification_report(y_pred, y_test))

lr_train_acc = round(lr.score(X_train, y_train) * 100, 2)

print('Training Accuracy: ', lr_train_acc)

lr_test_acc = round(lr.score(X_test, y_test) * 100, 2)

print('Testing Accuracy: ', lr_test_acc)
error_rate = []

for i in range(1,30):

    knn = KNeighborsClassifier(n_neighbors = i)

    knn.fit(X_train, y_train)

    pred_i = knn.predict(X_test)

    error_rate.append(np.mean(pred_i != y_test))

    

plt.figure(figsize = (8,6))

plt.plot(range(1,30), error_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)



print('Classification Report: \n', classification_report(y_pred, y_test))

knn_train_acc = round(knn.score(X_train, y_train) * 100, 2)

print('Training Accuracy: ', knn_train_acc)

knn_test_acc = round(knn.score(X_test, y_test) * 100, 2)

print('Testing Accuracy: ', knn_test_acc)
# We will use GridSearchCV to find best parameters

svc = SVC()

param_grid = {'C': [0.01, 0.1, 1 ,10 , 100], 'kernel':['linear', 'rbf'], 'gamma':[0.1, 1, 10, 100]}

gcv = GridSearchCV(estimator = svc, param_grid = param_grid, cv = 5, n_jobs=-1, refit=True)

gcv.fit(X_train, y_train)

gcv.best_params_
svc = SVC(C = 10, gamma = 0.1, kernel = 'rbf')

svc.fit(X_train,y_train)

y_pred = svc.predict(X_test)



print('Classification Report: \n', classification_report(y_pred, y_test))

svc_train_acc = round(svc.score(X_train, y_train) * 100, 2)

print('Training Accuracy: ', svc_train_acc)

svc_test_acc = round(svc.score(X_test, y_test) * 100, 2)

print('Testing Accuracy: ', svc_test_acc)
dt = DecisionTreeClassifier(max_depth = 6, min_samples_leaf = 2)

dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)



print('Classification Report: \n', classification_report(y_pred, y_test))

dt_train_acc = round(dt.score(X_train, y_train) * 100, 2)

print('Training Accuracy: ', dt_train_acc)

dt_test_acc = round(dt.score(X_test, y_test) * 100, 2)

print('Testing Accuracy: ', dt_test_acc)
rf = RandomForestClassifier()

param_grid = {'max_depth': [2, 4, 5, 6, 7, 8], 'criterion':['gini', 'entropy'], 'min_samples_leaf':[1, 2 ,4 ,6], 'max_features':['auto', 'log2'], 'n_estimators':[100,150,200]}

gcv = GridSearchCV(estimator=rf, param_grid=param_grid, cv = 5, n_jobs = -1)

gcv.fit(X_train, y_train)

gcv.best_params_
rf = RandomForestClassifier(max_depth = 8, min_samples_leaf = 6, n_estimators = 150)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)



print('Classification Report: \n', classification_report(y_pred, y_test))

rf_train_acc = round(rf.score(X_train, y_train) * 100, 2)

print('Training Accuracy: ', rf_train_acc)

rf_test_acc = round(rf.score(X_test, y_test) * 100, 2)

print('Testing Accuracy: ', rf_test_acc)
adb = AdaBoostClassifier(rf, n_estimators = 200)

adb.fit(X_train, y_train)

y_pred = adb.predict(X_test)



print('Classification Report: \n', classification_report(y_pred, y_test))

adb_train_acc = round(adb.score(X_train, y_train) * 100, 2)

print('Training Accuracy: ', adb_train_acc)

adb_test_acc = round(adb.score(X_test, y_test) * 100, 2)

print('Testing Accuracy: ', adb_test_acc)
gdb = GradientBoostingClassifier()

params = {'learning_rate':[0.01,0.1,1,10],'n_estimators':[100,150,200,300],'subsample':[0.6,0.8,1.0],'max_depth':[2,3,4,6],'min_samples_leaf':[1,2,4,6]}

gcv = GridSearchCV(estimator=gdb, param_grid=params, cv=5, n_jobs=-1)

gcv.fit(X_train, y_train)

gcv.best_params_
gdb = GradientBoostingClassifier(max_depth = 2, n_estimators = 300, subsample = 0.8)

gdb.fit(X_train, y_train)

y_pred = gdb.predict(X_test)



print('Classification Report: \n', classification_report(y_pred, y_test))

gdb_train_acc = round(gdb.score(X_train, y_train) * 100, 2)

print('Training Accuracy: ', gdb_train_acc)

gdb_test_acc = round(gdb.score(X_test, y_test) * 100, 2)

print('Testing Accuracy: ', gdb_test_acc)
xgbc = XGBClassifier(max_depth = 4)

xgbc.fit(X_train, y_train)

y_pred = xgbc.predict(X_test)



print('Classification Report: \n', classification_report(y_pred, y_test))

xgbc_train_acc = round(xgbc.score(X_train, y_train) * 100, 2)

print('Training Accuracy: ', xgbc_train_acc)

xgbc_test_acc = round(xgbc.score(X_test, y_test) * 100, 2)

print('Testing Accuracy: ', xgbc_test_acc)
x = ['Logistic Regression', 'KNN', 'SVC', 'Decision Tree','Random Forest','AdaBoost','Gradient Boosting','XGBoost']

y1 = [lr_train_acc, knn_train_acc, svc_train_acc, dt_train_acc, rf_train_acc, adb_train_acc, gdb_train_acc, xgbc_train_acc]

y2 = [lr_test_acc, knn_test_acc, svc_test_acc, dt_test_acc, rf_test_acc, adb_test_acc, gdb_test_acc, xgbc_test_acc]



trace1 = go.Bar(x = x, y = y1, name = 'Training Accuracy', marker = dict(color = 'forestgreen'))

trace2 = go.Bar(x = x, y = y2, name = 'Testing Accuracy', marker = dict(color = 'lawngreen'))

data = [trace1,trace2]

layout = go.Layout(title = 'Accuracy Plot', width = 750)

fig = go.Figure(data = data, layout = layout)

py.iplot(fig)
test_df['Fare'] = pd.to_numeric(test_df['Fare'])
test_df['Survived'] = rf.predict(test_df.drop(['PassengerId'], axis = 1))

test_df[['PassengerId', 'Survived']].to_csv('MySubmission.csv', index = False)