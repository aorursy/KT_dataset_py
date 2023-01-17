
import numpy as np 
import pandas as pd 

import os
print(os.listdir("../input"))

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid', color_codes=True)
palette = 'cubehelix' # good for bw printer

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

train_df.head()
train_df.info()
survived_sex = train_df[train_df['Survived']==1]['Sex'].value_counts()
dead_sex = train_df[train_df['Survived']==0]['Sex'].value_counts()
df = pd.DataFrame([survived_sex,dead_sex])
df.index = ['Survived','Dead']
df.plot(kind='bar',stacked=True, figsize=(10,5), color=['pink','blue'], title='Survival by the Sex')
df = train_df.copy()
df['Age'].fillna(df['Age'].median(), inplace=True)

figure = plt.figure(figsize=(10,5))
plt.hist([df[df['Survived']==1]['Age'],df[df['Survived']==0]['Age']], 
         stacked=True, color = ['g','r'], 
         bins = 30, label = ['Survived','Dead'])
plt.xlabel('Age')
plt.ylabel('Number of passengers')
plt.legend()
plt.title('Survival by Age')
df = train_df.copy()
df['Fare'].fillna(df['Fare'].median(), inplace=True)
figure = plt.figure(figsize=(10,5))
plt.hist(
            [df[df['Survived']==1]['Fare'],
            df[df['Survived']==0]['Fare']], 
            stacked=True, color = ['g','r'],
            bins = 30,
            label = ['Survived','Dead']
        )
plt.xlabel('Fare')
plt.ylabel('Number of passengers')
plt.legend()
plt.title('Survival by Ticket Price')
train_df.isnull().sum().sort_values(ascending=False)
test_df.isnull().sum().sort_values(ascending=False)

del train_df['Cabin']
# del train_df['Name']
# del train_df['PassengerId']
del train_df['Ticket']
del test_df['Cabin']
# del test_df['Name']
# del test_df['PassengerId']
del test_df['Ticket']

train_df=train_df.dropna(subset = ['Embarked'])
test_df=test_df.dropna(subset = ['Embarked'])

train_df['Fare'].fillna(train_df['Fare'].median(), inplace=True)
train_df['Age'].fillna(train_df['Age'].median(), inplace=True)
test_df['Fare'].fillna(train_df['Fare'].median(), inplace=True)
test_df['Age'].fillna(train_df['Age'].median(), inplace=True)
train_df.info()
train_df.isnull().sum().sort_values(ascending=False)
train_df['Title'] = train_df['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
test_df['Title'] = test_df['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
train_df['Title'].value_counts()
test_df['Title'].value_counts()
for i in range(len(test_df)):
    if test_df.loc[i,'Title'] not in list(set(list(train_df['Title']))):
        test_df.loc[i,'Title'] ='Don'
titles = (train_df['Title'].value_counts() < 10)
train_df['Title'] = train_df['Title'].apply(lambda x: 'other' if titles.loc[x] == True else x)
test_df['Title'] = test_df['Title'].apply(lambda x: 'other' if titles.loc[x] == True else x)

train_df['Title'].value_counts()
test_df['Title'].value_counts()
del train_df['Name']
del test_df['Name']
train_df.head()
train_df.info()
train_dummies = pd.get_dummies(train_df)
test_dummies = pd.get_dummies(test_df)
train_dummies.info()
train_dummies.head()
input_list = list(train_dummies.columns)
input_list.remove('Survived')
input_list.remove('PassengerId')
input_list
X_train = train_dummies[input_list]
Y_train = train_dummies['Survived']
X_test = test_dummies[input_list]
print(len(X_test))
# Y_test = test_dummies['Survived']

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
classifier = RandomForestClassifier(n_estimators = 80, criterion = 'entropy', max_depth = 6)
classifier.fit(X_train,Y_train)
Y_train_pred = classifier.predict(X_train)
Y_test_pred = classifier.predict(X_test)

confusion_matrix(Y_train, Y_train_pred)
len(X_test)
len(test_dummies['PassengerId'])
result = pd.DataFrame({'PassengerId' : test_dummies['PassengerId'],
                       'Survived' : Y_test_pred})

result.to_csv('result.csv',index=False)
