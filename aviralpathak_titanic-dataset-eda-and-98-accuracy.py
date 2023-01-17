# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/titanic/train.csv')

df_test = pd.read_csv('/kaggle/input/titanic/test.csv')
df_combine = pd.concat([df,df_test])

df_combine.reset_index(drop=True, inplace=True)
df.head()
df.info()
total = df.isnull().sum().sort_values(ascending = False)

percent = round(df.isnull().sum().sort_values(ascending = False) * 100 /len(df),2)

pd.concat([total, percent], axis=1, keys=['Total','Percent'])
df1 = df.drop(columns=["Cabin","PassengerId","Name","Ticket"])



df1['Age'].fillna((df1['Age'].mean()), inplace=True)



df1["Embarked"].fillna("S", inplace = True)
df1['Family'] = df1['SibSp'] + df1['Parch']
df1.drop(['SibSp','Parch'], axis=1, inplace=True)
df1.head()
df_test1 = df_test.drop(columns=["Cabin","PassengerId","Name","Ticket"])



df_test1['Age'].fillna((df_test1['Age'].mean()), inplace=True)



df_test1['Fare'].fillna((df_test1['Fare'].mean()), inplace=True)
df_test1
df_test1['Family'] = df_test1['SibSp'] + df_test1['Parch']

df_test1.drop(['SibSp','Parch'], axis=1, inplace=True)
df_test1.info()
plt.figure(figsize = (18,8))

sns.catplot(x = 'Age', y= 'Fare', data = df1, kind='point', aspect=4);

plt.show()
fig, saxis = plt.subplots(1, 3,figsize=(16,8))

sns.boxplot(y=df1['Fare'], ax = saxis[0])

sns.boxplot(y=df1['Age'], ax = saxis[1])

sns.boxplot(y=df1['Family'], ax = saxis[2])
fig, saxis = plt.subplots(1, 3,figsize=(16,8))



sns.countplot(df1['Pclass'], hue = df1['Survived'], ax = saxis[0])

sns.countplot(df1['Embarked'], hue = df1['Survived'], ax = saxis[1])

sns.countplot(df1['Sex'], hue = df1['Survived'], ax = saxis[2])
plt.figure(figsize=[16,15])



plt.subplot(231)

plt.hist(x = [df1[df1['Survived']==1]['Age'], df1[df1['Survived']==0]['Age']],stacked=True)

plt.xlabel('Age (Years)')

plt.ylabel('No. of Passengers')



plt.subplot(232)

plt.hist(x = [df1[df1['Survived']==1]['Fare'], df1[df1['Survived']==0]['Fare']],stacked=True)

plt.xlabel('Fare')

plt.ylabel('No. of Passengers')



plt.subplot(233)

plt.hist(x = [df1[df1['Survived']==1]['Family'], df1[df1['Survived']==0]['Family']],stacked=True)

plt.xlabel('Family Members')

plt.ylabel('No. of Passengers')
h = sns.FacetGrid(df1, row = 'Sex', col = 'Pclass', hue = 'Survived')

h.map(plt.hist, 'Age', alpha = .75)

h.add_legend()
grid = sns.FacetGrid(df1, row='Embarked', col='Survived', size=2.2, aspect=1.6)

grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)

grid.add_legend()
#categorical data

df_cat = df1[['Sex','Embarked']]

df_test_cat = df_test1[['Sex','Embarked']]
df_cat = pd.get_dummies(df_cat)

df_test_cat = pd.get_dummies(df_test_cat)
df_cat
#numerical data

df_num = df1[['Age','Fare']]

df_test_num = df_test1[['Age','Fare']]
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_num[['Age','Fare']] = scaler.fit_transform(df_num[['Age','Fare']])

df_test_num[['Age','Fare']] = scaler.fit_transform(df_test_num[['Age','Fare']])
df_num
df_test
df2 = pd.concat([df_num,df_cat,df1[['Survived','Pclass','Family']]],axis=1)

df_test2 = pd.concat([df_test_num,df_test_cat,df_test1[['Pclass','Family']]],axis=1)
from sklearn.model_selection import train_test_split
X_train = df2.drop("Survived", axis = 1)

y_train = df2["Survived"]

X_test= df_test2
from sklearn.tree import DecisionTreeClassifier

algo = DecisionTreeClassifier()

algo.fit(X_train, y_train)
model = DecisionTreeClassifier() 

model.fit(X_train, y_train)

y_pred_dt = model.predict(X_test) 

model.score(X_train,y_train)
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors = 3) 

model.fit(X_train, y_train)  

y_pred_knn = model .predict(X_test)  

model.score(X_train,y_train)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

model.fit(X_train,y_train)

y_pred_lr=model.predict(X_test)

model.score(X_train,y_train)