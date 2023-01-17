# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/titanic/train.csv")

test_df = pd.read_csv("../input/titanic/test.csv")

df.head()


print('Cabin null values:',df['Cabin'].isna().sum())

print('Pclass null values:',df['Pclass'].isna().sum())

print('Name null values:',df['Name'].isna().sum())

print('Ticket null values:',df['Ticket'].isna().sum())

print('Fare null values:',df['Fare'].isna().sum())

print('Survived null values:',df['Survived'].isna().sum())

print('Embarked null values:',df['Embarked'].isna().sum())

print('SibSp null values:',df['SibSp'].isna().sum())

print('Parch null values:',df['Parch'].isna().sum())

print('Age null values:',df['Age'].isna().sum())

df.dropna(subset=['Embarked'],inplace=True)
df.describe()
fig,ax1=plt.subplots(1,1,figsize=(10,10))

labels = ['Not Survived','Survived']

v=df['Survived']

ax1.pie(v.value_counts().loc[[0,1]],labels = labels,autopct='%1.1f%%',)

df['cSex']=[1 if c=='female' else 0 for c in df['Sex']]

df['cEmbarked'] = df.Embarked.map( {'C': 0, 'Q': 1, 'S': 2} ).astype(int)



test_df['cSex']=[1 if c=='female' else 0 for c in test_df['Sex']]

test_df['cEmbarked'] = test_df.Embarked.map( {'C': 0, 'Q': 1, 'S': 2} ).astype(int)
corr = df.corr()

fig, ax = plt.subplots(figsize=(10,10))

sns.heatmap(corr, 

        xticklabels=corr.columns,

        yticklabels=corr.columns, ax = ax)
fig,((ax1,ax2),(ax3,ax4))=plt.subplots(2,2,figsize=(10,10))

labels = ['male','female']

v=df[df['Survived']==1]['cSex']

ax1.pie(v.value_counts().loc[[0,1]],labels = labels,autopct='%1.1f%%',)

ax1.set_title('Survived Gender')



dead_sex=df[df['Survived']==0]['cSex']

ax2.pie(dead_sex.value_counts().loc[[0,1]],labels=labels,autopct='%1.1f%%',)

ax2.set_title('Not Survived Gender')



ax3.pie(df[df['cSex']==0]['Survived'].value_counts().loc[[0,1]],labels=['Not Survived','Survived'],autopct='%1.1f%%',)

ax3.set_title('Male')



ax4.pie(df[df['cSex']==1]['Survived'].value_counts().loc[[0,1]],labels=['Not Survived','Survived'],autopct='%1.1f%%',)

ax4.set_title('Female')

fig.suptitle('Gender Analysis')

plt.show()



fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,10))

v=df[df['Survived']==1]['Pclass']

ax1.pie(v.value_counts().loc[[1,2,3]],labels=['1st','2nd','3rd'],autopct='%1.1f%%',)

ax1.set_title('Survived Class')



v=df[df['Survived']==0]['Pclass']

ax2.pie(v.value_counts().loc[[1,2,3]],labels=['1st','2nd','3rd'],autopct='%1.1f%%',)

ax2.set_title('Not Survived Class')



fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(12,5))

ax1.pie(df[df['Pclass']==1]['Survived'].value_counts().loc[[0,1]],labels=['Not Survived','Survived'],autopct='%1.1f%%',)

ax1.set_title('1st Class')



ax2.pie(df[df['Pclass']==2]['Survived'].value_counts().loc[[0,1]],labels=['Not Survived','Survived'],autopct='%1.1f%%',)

ax2.set_title('2nd Class')



ax3.pie(df[df['Pclass']==3]['Survived'].value_counts().loc[[0,1]],labels=['Not Survived','Survived'],autopct='%1.1f%%',)

ax3.set_title('3rd Class')





fig.suptitle('PClass')

plt.show()
df_nn = df[pd.notnull(df['Age'])]

import seaborn as sns

sns.violinplot(df_nn['Age'], df_nn['Sex'], cut=0.) #Variable Plot

sns.despine()
fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(12,5))

ax1.pie(df[df['cEmbarked']==0]['Survived'].value_counts().loc[[0,1]],labels=['Not Survived','Survived'],autopct='%1.1f%%',)

ax1.set_title('Cherbourg')



ax2.pie(df[df['cEmbarked']==1]['Survived'].value_counts().loc[[0,1]],labels=['Not Survived','Survived'],autopct='%1.1f%%',)

ax2.set_title('Queenstown')



ax3.pie(df[df['cEmbarked']==2]['Survived'].value_counts().loc[[0,1]],labels=['Not Survived','Survived'],autopct='%1.1f%%',)

ax3.set_title('Southamtpn')



plt.show()
fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(12,5))

ax1.pie(df[df['cEmbarked']==0]['Pclass'].value_counts().loc[[1,2,3]],labels=['1st','2nd','3rd'],autopct='%1.1f%%',)

ax1.set_title('Cherbourg')



ax2.pie(df[df['cEmbarked']==1]['Pclass'].value_counts().loc[[1,2,3]],labels=['1st','2nd','3rd'],autopct='%1.1f%%',)

ax2.set_title('Queenstown')



ax3.pie(df[df['cEmbarked']==2]['Pclass'].value_counts().loc[[1,2,3]],labels=['1st','2nd','3rd'],autopct='%1.1f%%',)

ax3.set_title('Southampton')



plt.show()
df[['Sex','Age']].groupby('Sex').mean()
df[['Pclass','Age']].groupby('Pclass').mean()
grid = sns.FacetGrid(df, row='Pclass', col='Sex', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend()
grid = sns.FacetGrid(df, row='Pclass', col='Survived', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend()
#prendendo spunto dal Kernel https://www.kaggle.com/startupsci/titanic-data-science-solutions possiamo fare come segue:

combine = [df,test_df]

guess_ages = np.zeros((2,3))

for dataset in combine:

    for i in range(0, 2):

        for j in range(0, 3):

            guess_df = dataset[(dataset['cSex'] == i) & (dataset['Pclass'] == j+1)]['Age'].dropna()

            age_guess = guess_df.median()

            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

        

    for i in range(0, 2):

        for j in range(0, 3):

            dataset.loc[ (dataset.Age.isnull()) & (dataset.cSex == i) & (dataset.Pclass == j+1),\

                    'Age'] = guess_ages[i,j]



    dataset['Age'] = dataset['Age'].astype(int)



df.head()
pd.DataFrame(guess_ages,columns=['1stClass','2ndClass','3rdClass'],index=['male','female'])
from sklearn import tree

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier,plot_tree

from sklearn.tree import export_graphviz

from sklearn.model_selection import cross_val_score



train_features = df[['Pclass','cSex','cEmbarked','Age','SibSp','Parch']]



train_target = df[['Survived']]
import matplotlib.pyplot as plt

from subprocess import call

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

    

    

def display_tree(feats, p1):

    X_train, X_test, y_train, y_test = train_test_split(feats, train_target, test_size=0.20, random_state=42)

    model = tree.DecisionTreeClassifier()

    model.fit(X_train,y_train)

    results = model.predict(X_test)

    acc = accuracy_score(results,y_test)

    export_graphviz(model, out_file='tree.dot',

                feature_names = p1,

                class_names=['Non Sopravvissuto','Sopravvissuto'],

                rounded = True, proportion = False, 

                precision = 2, filled = True, impurity = False)

    call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])



    plt.figure(figsize = (14, 18))

    plt.imshow(plt.imread('tree.png'))

    plt.axis('off');

    plt.show();

    print(acc)

display_tree(df[['Pclass']],['Classe'])
display_tree(df[['cSex','Pclass']],['Uomo','Classe'])
display_tree(df[['cSex','Pclass','Age']],['Uomo','Classe','Età'])
for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



pd.crosstab(df['Survived'], df['Title'])
for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    

df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in combine:

    dataset['cTitle'] = dataset['Title'].map(title_mapping)

    dataset['cTitle'] = dataset['cTitle'].fillna(0)



df.head()
for dataset in combine:    

    dataset.loc[ dataset['Age'] <= 5, 'cAge'] = 0

    dataset.loc[(dataset['Age'] > 5) & (dataset['Age'] <= 10), 'cAge'] = 1

    dataset.loc[(dataset['Age'] > 10) & (dataset['Age'] <= 20), 'cAge'] = 2

    dataset.loc[(dataset['Age'] > 20) & (dataset['Age'] <= 35), 'cAge'] = 3

    dataset.loc[(dataset['Age'] > 35) & (dataset['Age'] <= 45), 'cAge'] = 4

    dataset.loc[(dataset['Age'] > 45) & (dataset['Age'] <= 55), 'cAge'] = 5

    dataset.loc[ dataset['Age'] > 55, 'cAge'] = 6

df.head()
for dataset in combine:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] 

    

for dataset in combine:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1



df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
for dataset in combine:

    dataset.loc[ dataset['Fare'] <= 7.91, 'cFare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'cFare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'cFare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'cFare'] = 3

    #dataset['cFare'] = dataset['cFare'].astype(int)



#df = df.drop(['FareBand'], axis=1)

combine = [df, test_df]

    

df.head(10)
for dataset in combine:

    dataset['Age*Class'] = dataset.cAge * dataset.Pclass



df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)
from sklearn.metrics import recall_score

from sklearn.metrics import precision_score



feats =df[['cSex','Pclass','IsAlone','cTitle','Age*Class','FamilySize','cAge']]

X_train, X_test, y_train, y_test = train_test_split(feats, train_target, test_size=0.20, random_state=42)

model = tree.DecisionTreeClassifier()

model.fit(X_train,y_train)

res = model.predict(X_test)

acc = accuracy_score(res,y_test)

recall = recall_score(res,y_test)

precision = precision_score(res,y_test)

print(acc,recall,precision)    
feats =df[['cSex','Pclass','cAge','IsAlone','cTitle','Age*Class','FamilySize']]

X_train, X_test, y_train, y_test = train_test_split(feats, train_target, test_size=0.20, random_state=42)

model = SVC()

model.fit(X_train,y_train.values.ravel())

res = model.predict(X_test)

acc = accuracy_score(res,y_test.values.ravel())

recall = recall_score(res,y_test)

precision = precision_score(res,y_test)

print(acc,recall,precision) 
feats =df[['cSex','Pclass','cAge','IsAlone','cTitle','Age*Class','FamilySize']]

X_train, X_test, y_train, y_test = train_test_split(feats, train_target, test_size=0.20, random_state=42)

model = KNeighborsClassifier(n_neighbors = 3)

model.fit(X_train,y_train.values.ravel())

res = model.predict(X_test)

acc = accuracy_score(res,y_test)

recall = recall_score(res,y_test)

precision = precision_score(res,y_test)

print(acc,recall,precision)     
feats =df[['cSex','Pclass','cAge','IsAlone','cTitle','Age*Class','FamilySize']]

X_train, X_test, y_train, y_test = train_test_split(feats, train_target, test_size=0.20, random_state=42)

model = RandomForestClassifier(n_estimators=100)

model.fit(X_train,y_train.values.ravel())

res = model.predict(X_test)

acc = accuracy_score(res,y_test)

recall = recall_score(res,y_test)

precision = precision_score(res,y_test)

print(acc,recall,precision) 
acc = accuracy_score([0]*len(y_test),y_test)

recall = recall_score([0]*len(y_test),y_test)

print(acc,recall)
def dummy(X):

    return [0 if i ==0 else 1 for i in X['cSex']]

        
acc = accuracy_score(dummy(X_test),y_test)

recall = recall_score(dummy(X_test),y_test)

precision = precision_score(dummy(X_test),y_test)

print(acc,recall,precision) 


