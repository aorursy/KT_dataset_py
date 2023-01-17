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
import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import re



import warnings

warnings.filterwarnings('ignore')



from sklearn.model_selection import  GridSearchCV,train_test_split

from sklearn.metrics import classification_report,confusion_matrix

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
train_df = pd.read_csv("../input/titanic/train.csv")

print(train_df.shape)

train_df.head()
test_df = pd.read_csv('../input/titanic/test.csv')

print(test_df.shape)

test_df.head()
train_df.info()
train_df.info()
train_df.describe()
sns.heatmap(train_df.corr(),annot=True)
survived = train_df[(train_df["Sex"]=="female") & (train_df["Survived"]==1)]

print(len(survived.index))
print([train_df.groupby("Sex")["Survived"].value_counts(normalize = True)])
pivot_table = train_df.pivot_table(index="Sex",values="Survived")

pivot_table.plot.bar()

plt.show()
def bar_chart(feature):

    survived = train_df[train_df['Survived']==1][feature].value_counts()

    dead = train_df[train_df['Survived']==0][feature].value_counts()

    df = pd.DataFrame([survived,dead])

    df.index = ['Survived','Dead']

    df.plot(kind='bar',stacked=True, figsize=(10,5))
df = [train_df,test_df]   #Combininng Train and Test Dataset

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
bar_chart('Title')
#Drop unnecessary feature from dataset

train_df.drop('Name', axis=1, inplace=True)

test_df.drop('Name', axis=1, inplace=True)
train_df.head(1)
sex_mapping = {"male":0,"female":1}

for data in df:

    data['Sex']=data['Sex'].map(sex_mapping)
bar_chart('Sex')
print('Missing Values in Age column: ',177/len(train_df['Age'])*100)

print('Missing Values in Cabin column: ',687/len(train_df['Cabin'])*100)

print('Missing Values in Embarked column: ',2/len(train_df['Embarked'])*100)
fig, (ax1, ax2) = plt.subplots(ncols = 2, figsize = (15,5))

sns.heatmap(train_df.isnull(), cmap = 'coolwarm', ax = ax1)

sns.heatmap(test_df.isnull(), cmap = 'mako_r', ax = ax2)
# fill missing age with median age for each title (Mr, Mrs, Miss, Others)

train_df["Age"].fillna(train_df.groupby("Title")["Age"].transform("median"), inplace=True)

test_df["Age"].fillna(test_df.groupby("Title")["Age"].transform("median"), inplace=True)
train_df.info()
test_df.info()
# fill missing age with median age for each title (Mr, Mrs, Miss, Rare)

train_df["Age"].fillna(train_df.groupby("Title")["Age"].transform("median"), inplace=True)

test_df["Age"].fillna(test_df.groupby("Title")["Age"].transform("median"), inplace=True)
train_df.info()
for data in df:

    data.loc[ data['Age'] <= 16, 'Age'] = 0,

    data.loc[(data['Age'] > 16) & (data['Age'] <= 26), 'Age'] = 1,

    data.loc[(data['Age'] > 26) & (data['Age'] <= 36), 'Age'] = 2,

    data.loc[(data['Age'] > 36) & (data['Age'] <= 62), 'Age'] = 3,

    data.loc[ data['Age'] > 62, 'Age'] = 4
train_df.head(5)
bar_chart('Age')
Pclass1 = train_df[train_df['Pclass']==1]['Embarked'].value_counts()

Pclass2 = train_df[train_df['Pclass']==2]['Embarked'].value_counts()

Pclass3 = train_df[train_df['Pclass']==3]['Embarked'].value_counts()

df = pd.DataFrame([Pclass1, Pclass2, Pclass3])

df.index = ['1st class','2nd class', '3rd class']

df.plot(kind='bar',stacked=True, figsize=(10,5))
train_df['Embarked'].fillna('S', inplace = True)



label = {'S':1, 'C':2, 'Q':3}

train_df.replace({'Embarked':label}, inplace = True)

test_df.replace({'Embarked':label}, inplace = True)
bar_chart('Embarked')
test_df['Fare'].fillna(test_df['Fare'].median(), inplace = True)

train_df['Fare'].fillna(train_df['Fare'].median(), inplace = True)


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
train_df["CabinBool"] = (train_df["Cabin"].notnull().astype('int'))

test_df["CabinBool"] = (test_df["Cabin"].notnull().astype('int'))



#calculate percentages of CabinBool vs. survived

print("Percentage of CabinBool = 1 who survived:", train_df["Survived"][train_df["CabinBool"] == 1].value_counts(normalize = True)[1]*100)



print("Percentage of CabinBool = 0 who survived:", train_df["Survived"][train_df["CabinBool"] == 0].value_counts(normalize = True)[1]*100)

#draw a bar plot of CabinBool vs. survival

sns.barplot(x="CabinBool", y="Survived", data=train_df)

plt.show()
train_df["FamilySize"] = train_df["SibSp"] + train_df["Parch"] + 1

test_df["FamilySize"] = test_df["SibSp"] + test_df["Parch"] + 1
train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df.head(3)
test_df.head(3)
train_df.drop(['PassengerId','Ticket', 'Age', 'SibSp', 'Parch','Cabin'], axis = 1, inplace = True)

test_df.drop(['Ticket', 'Age', 'SibSp', 'Parch','Cabin'], axis = 1, inplace = True)
train_df.head(5)
X = train_df.drop('Survived', axis = 1)

y = train_df['Survived']
from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
lr = LogisticRegression()

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)



print('Classification Report: \n', classification_report(y_pred, y_test))

lr_train_acc = round(lr.score(X_train, y_train) * 100, 2)

print('Training Accuracy: ', lr_train_acc)

lr_test_acc = round(lr.score(X_test, y_test) * 100, 2)

print('Testing Accuracy: ', lr_test_acc)
error_rate= []

for i in range(1,30):

    knn = KNeighborsClassifier(n_neighbors = i)

    knn.fit(X_train, y_train)

    pred_i = knn.predict(X_test)

    error_rate.append(np.mean(pred_i != y_test))

    

plt.figure(figsize = (8,6))

plt.plot(range(1,30), error_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train,y_train)

y_pred=knn.predict(X_test)



print('Classification Report: \n',classification_report(y_pred,y_test))

knn_train_acc = round(knn.score(X_train,y_train)*100,2)

print('Training Accuracy:',knn_train_acc)

knn_test_acc = round(knn.score(X_test,y_test)*100,2)

print('Testing Accuracy:',knn_test_acc)
svc = SVC()

svc.fit(X_train,y_train)

y_pred=svc.predict(X_test)



print('Classification Report: \n',classification_report(y_pred,y_test))

svc_train_acc = round(svc.score(X_train,y_train)*100,2)

print('Training Accuracy:',svc_train_acc)

svc_test_acc = round(svc.score(X_test,y_test)*100,2)

print('Testing Accuracy:',svc_test_acc)
dt =DecisionTreeClassifier(min_samples_split=70,min_samples_leaf=10)

dt.fit(X_train,y_train)

prediction = dt.predict(X_test)



print('Classification Report: \n',classification_report(y_pred,y_test))

dt_train_acc = round(dt.score(X_train,y_train)*100,2)

print('Training Accuracy:',dt_train_acc)

dt_test_acc = round(dt.score(X_test,y_test)*100,2)

print('Testing Accuracy:',dt_test_acc)
rf = RandomForestClassifier(n_estimators=100)

rf.fit(X_train,y_train)

rf_pred = rf.predict(X_test)



print('Classification Report: \n',classification_report(y_pred,y_test))

rf_train_acc = round(rf.score(X_train,y_train)*100,2)

print('Training Accuracy:',rf_train_acc)

rf_test_acc = round(rf.score(X_test,y_test)*100,2)

print('Testing Accuracy:',rf_test_acc)
test_df['Fare'] = pd.to_numeric(test_df['Fare'])
test_df['Survived'] = rf.predict(test_df.drop(['PassengerId'], axis = 1))

test_df[['PassengerId', 'Survived']].to_csv('MySubmission.csv', index = False)