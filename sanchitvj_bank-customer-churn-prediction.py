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

sns.set_style('darkgrid')

%matplotlib inline



import plotly.graph_objs as go

import plotly.offline as py

py.init_notebook_mode(connected=True)



import warnings

warnings.filterwarnings('ignore')
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler



from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from xgboost import XGBClassifier
df = pd.read_csv('../input/will-customer-leave-bank/ML_TASK_CSV.csv')
df.head()
df.shape
df.describe()
df.info()
#Correlation Graph

plt.figure(figsize = (10,8))

sns.heatmap(df.corr(), annot = True)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

females = df[df['Gender'] == 'Female']

males = df[df['Gender'] == 'Male']



ax = sns.distplot(females[females['Exited'] == 1].Age, bins=30, label='Exited', ax=axes[0], hist_kws = {'edgecolor':'white'})

ax = sns.distplot(females[females['Exited'] == 0].Age, bins=30, label='Not Exited', ax=axes[0], hist_kws = {'edgecolor':'white'})

ax.legend()

ax.set_title('Female')

ax = sns.distplot(males[males['Exited'] == 1].Age, bins=30, label='Exited', ax=axes[1], hist_kws = {'edgecolor':'white'})

ax = sns.distplot(males[males['Exited'] == 0].Age, bins=30, label='Not Exited', ax=axes[1], hist_kws = {'edgecolor':'white'})

ax.legend()

ax.set_title('Male')
sns.jointplot(x = 'Age', y = 'EstimatedSalary', data = df, kind = 'hex', color = 'green')
sns.jointplot(x = 'Age', y = 'CreditScore', data = df, kind = 'hex', color = 'cyan')
age = []

for i in df['Age']:

    if i <= 33:

        age.append(1)

    elif i >33 and i <= 40:

        age.append(2)

    elif i > 40:

        age.append(3)

        

df['Age'] = age
df_1 = df[df['Age'] == 1]

df_2 = df[df['Age'] == 2]

df_3 = df[df['Age'] == 3]

df_1 = df_1['Exited'].value_counts()

df_2 = df_2['Exited'].value_counts()

df_3 = df_3['Exited'].value_counts()



trace1 = go.Bar(x = df_1.index[::-1], y = df_1.values[::-1], name = 'Age(18-33)', marker = dict(color = 'cadetblue'))

trace2 = go.Bar(x = df_2.index[::-1], y = df_2.values[::-1], name = 'Age(34-40)', marker = dict(color = 'teal'))

trace3 = go.Bar(x = df_3.index[::-1], y = df_3.values[::-1], name = 'Age(40-92)', marker = dict(color = 'seagreen'))

data = [trace1, trace2, trace3]

layout = go.Layout(height = 400, width = 700, title = 'Age Distribution')

fig = go.Figure(data = data, layout = layout)

py.iplot(fig)
df_1 = df[df['Exited'] == 1]

df_2 = df[df['Exited'] == 0]

df_1 = df_1['Tenure'].value_counts()

df_2 = df_2['Tenure'].value_counts()



trace1 = go.Bar(x = df_1.index[::-1], y = df_1.values[::-1], name = 'Exited', marker = dict(color = 'cadetblue'))

trace2 = go.Bar(x = df_2.index[::-1], y = df_2.values[::-1], name = 'Not Exited', marker = dict(color = 'teal'))

data = [trace1, trace2]

layout = go.Layout(height = 400, width = 700, title = 'Tenure Distribution')

fig = go.Figure(data = data, layout = layout)

py.iplot(fig)
df_1 = df[df['Gender'] == 'Male']

df_2 = df[df['Gender'] == 'Female']

df_1 = df_1['Exited'].value_counts()

df_2 = df_2['Exited'].value_counts()



trace1 = go.Bar(x = df_1.index[::-1], y = df_1.values[::-1], name = 'Male', marker = dict(color = 'lightseagreen'))

trace2 = go.Bar(x = df_2.index[::-1], y = df_2.values[::-1], name = 'Female', marker = dict(color = 'crimson'))

data = [trace1, trace2]

layout = go.Layout(height = 400, width = 700, title = 'Gender Distribution')

fig = go.Figure(data = data, layout = layout)

py.iplot(fig)
labels = {'Female':0, 'Male':1}

df.replace({'Gender':labels}, inplace = True)
df_1 = df[df['Exited'] == 1]

df_2 = df[df['Exited'] == 0]

df_1 = df_1['Geography'].value_counts()

df_2 = df_2['Geography'].value_counts()



trace1 = go.Bar(x = df_1.index[::-1], y = df_1.values[::-1], name = 'Exited', marker = dict(color = 'indigo'))

trace2 = go.Bar(x = df_2.index[::-1], y = df_2.values[::-1], name = 'Not Exited', marker = dict(color = 'green'))

data = [trace1, trace2]

layout = go.Layout(height = 400, width = 700, title = 'Geography Distribution')

fig = go.Figure(data = data, layout = layout)

py.iplot(fig)
labels = {'Spain':1, 'France':2, 'Germany':3}

df.replace({'Geography':labels}, inplace = True)
df_1 = df[df['Balance'] == 0.00]

df_2 = df[df['Balance'] != 0.00]

df_1 = df_1['Exited'].value_counts()

df_2 = df_2['Exited'].value_counts()



trace1 = go.Bar(x = df_1.index[::-1], y = df_1.values[::-1], name = 'Exited', marker = dict(color = 'peru'))

trace2 = go.Bar(x = df_2.index[::-1], y = df_2.values[::-1], name = 'Not Exited', marker = dict(color = 'darkred'))

data = [trace1, trace2]

layout = go.Layout(height = 400, width = 700, title = 'Balance Distribution')

fig = go.Figure(data = data, layout = layout)

py.iplot(fig)
balance = []

for i in df['Balance']:

    if i == 0.0:

        balance.append(0)

    else:

        balance.append(1)

        

df['Balance'] = balance
credit = []

for i in df['CreditScore']:

    if i < 600:

        credit.append(1)

    elif i >= 600 and i < 700:

        credit.append(0)

    elif i >= 700:

        credit.append(1)

        

df['CreditScore'] = credit
df_1 = df[df['CreditScore'] == 1]

df_2 = df[df['CreditScore'] == 0]

df_1 = df_1['Exited'].value_counts()

df_2 = df_2['Exited'].value_counts()



trace1 = go.Bar(x = df_1.index[::-1], y = df_1.values[::-1], name = 'CScore(<=600 & >700)', marker = dict(color = 'chartreuse'))

trace2 = go.Bar(x = df_2.index[::-1], y = df_2.values[::-1], name = 'CScore(601-700)', marker = dict(color = 'coral'))

data = [trace1, trace2]

layout = go.Layout(height = 400, width = 700, title = 'CreditScore Distribution')

fig = go.Figure(data = data, layout = layout)

py.iplot(fig)
df_1 = df[df['Exited'] == 1]

df_2 = df[df['Exited'] == 0]

df_1 = df_1['NumOfProducts'].value_counts()

df_2 = df_2['NumOfProducts'].value_counts()



trace1 = go.Bar(x = df_1.index[::-1], y = df_1.values[::-1], name = 'Exited', marker = dict(color = 'deeppink'))

trace2 = go.Bar(x = df_2.index[::-1], y = df_2.values[::-1], name = 'Not Exited', marker = dict(color = 'yellow'))

data = [trace1, trace2]

layout = go.Layout(height = 400, width = 700, title = 'NumOfProducts Distribution')

fig = go.Figure(data = data, layout = layout)

py.iplot(fig)
df.drop(['RowNumber', 'CustomerId', 'Surname'], axis = 1, inplace = True)
#RobustScaler works best and effectively on continuous Estimated Salary data

scaler = RobustScaler()

df[['EstimatedSalary']] = scaler.fit_transform(df[['EstimatedSalary']])

df.head()
X = df.drop('Exited', axis = 1)

y = df['Exited']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 123)
lr = LogisticRegression()

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)



print('Classification Report: \n', classification_report(y_pred, y_test))

lr_train_acc = round(lr.score(X_train, y_train) * 100, 2)

print('Training Accuracy: ', lr_train_acc)

lr_test_acc = round(lr.score(X_test, y_test) * 100, 2)

print('Testing Accuracy: ', lr_test_acc)

print('Confusion Matrix: \n', confusion_matrix(y_test, y_pred))
knn = KNeighborsClassifier()

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)



print('Classification Report: \n', classification_report(y_pred, y_test))

knn_train_acc = round(knn.score(X_train, y_train) * 100, 2)

print('Training Accuracy: ', knn_train_acc)

knn_test_acc = round(knn.score(X_test, y_test) * 100, 2)

print('Testing Accuracy: ', knn_test_acc)

print('Confusion Matrix: \n', confusion_matrix(y_test, y_pred))
svc = SVC(C = 1, gamma = 1)

svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)



print('Classification Report: \n', classification_report(y_pred, y_test))

svc_train_acc = round(svc.score(X_train, y_train) * 100, 2)

print('Training Accuracy: ', svc_train_acc)

svc_test_acc = round(svc.score(X_test, y_test) * 100, 2)

print('Testing Accuracy: ', svc_test_acc)

print('Confusion Matrix: \n', confusion_matrix(y_test, y_pred))
dt = DecisionTreeClassifier(max_depth = 4)

dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)



print('Classification Report: \n', classification_report(y_pred, y_test))

dt_train_acc = round(dt.score(X_train, y_train) * 100, 2)

print('Training Accuracy: ', dt_train_acc)

dt_test_acc = round(dt.score(X_test, y_test) * 100, 2)

print('Testing Accuracy: ', dt_test_acc)

print('Confusion Matrix: \n', confusion_matrix(y_test, y_pred))
rf = RandomForestClassifier(n_estimators = 100, max_depth = 10, n_jobs = -1)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)



print('Classification Report: \n', classification_report(y_pred, y_test))

rf_train_acc = round(rf.score(X_train, y_train) * 100, 2)

print('Training Accuracy: ', rf_train_acc)

rf_test_acc = round(rf.score(X_test, y_test) * 100, 2)

print('Testing Accuracy: ', rf_test_acc)

print('Confusion Matrix: \n', confusion_matrix(y_test, y_pred))
adb = AdaBoostClassifier(n_estimators = 300)

adb.fit(X_train, y_train)

y_pred = adb.predict(X_test)



print('Classification Report: \n', classification_report(y_pred, y_test))

adb_train_acc = round(adb.score(X_train, y_train) * 100, 2)

print('Training Accuracy: ', adb_train_acc)

adb_test_acc = round(adb.score(X_test, y_test) * 100, 2)

print('Testing Accuracy: ', adb_test_acc)

print('Confusion Matrix: \n', confusion_matrix(y_test, y_pred))
gdb = GradientBoostingClassifier(n_estimators = 200, subsample = 0.8)

gdb.fit(X_train, y_train)

y_pred = gdb.predict(X_test)



print('Classification Report: \n', classification_report(y_pred, y_test))

gdb_train_acc = round(gdb.score(X_train, y_train) * 100, 2)

print('Training Accuracy: ', gdb_train_acc)

gdb_test_acc = round(gdb.score(X_test, y_test) * 100, 2)

print('Testing Accuracy: ', gdb_test_acc)

print('Confusion Matrix: \n', confusion_matrix(y_test, y_pred))
xgbc = XGBClassifier(max_depth = 3)

xgbc.fit(X_train, y_train)

y_pred = xgbc.predict(X_test)



print('Classification Report: \n', classification_report(y_pred, y_test))

xgbc_train_acc = round(xgbc.score(X_train, y_train) * 100, 2)

print('Training Accuracy: ', xgbc_train_acc)

xgbc_test_acc = round(xgbc.score(X_test, y_test) * 100, 2)

print('Testing Accuracy: ', xgbc_test_acc)

print('Confusion Matrix: \n', confusion_matrix(y_test, y_pred))
x = ['Logistic Regression', 'KNN', 'SVC', 'Decision Tree','Random Forest','AdaBoost','Gradient Boosting','XGBoost']

y1 = [lr_train_acc, knn_train_acc, svc_train_acc, dt_train_acc, rf_train_acc, adb_train_acc, gdb_train_acc, xgbc_train_acc]

y2 = [lr_test_acc, knn_test_acc, svc_test_acc, dt_test_acc, rf_test_acc, adb_test_acc, gdb_test_acc, xgbc_test_acc]



trace1 = go.Bar(x = x, y = y1, name = 'Training Accuracy', marker = dict(color = 'forestgreen'))

trace2 = go.Bar(x = x, y = y2, name = 'Testing Accuracy', marker = dict(color = 'lawngreen'))

data = [trace1,trace2]

layout = go.Layout(title = 'Accuracy Plot', width = 750)

fig = go.Figure(data = data, layout = layout)

py.iplot(fig)