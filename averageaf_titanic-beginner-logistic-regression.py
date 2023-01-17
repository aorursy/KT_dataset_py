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
train = pd.read_csv('/kaggle/input/train.csv')

train.head()
temp = train
missingdata = train.isnull().sum()

missingdata = missingdata[missingdata >0]

missingdata.sort_values(inplace=True)

mdplt = missingdata.plot.bar();

mdplt.set_title('Null Entries per Category');
train.drop('Cabin',inplace=True, axis=1)
train.drop('PassengerId',inplace=True, axis=1)

train.drop('Ticket',inplace=True, axis=1)
plt.figure(figsize=(10,12))

sns.boxplot('Pclass','Age',data=train);
print("The mean before is ", train.Age.mean())

print(train.groupby('Pclass').median()['Age'])
def fill_age(columns):

    Age = columns[0]

    Pclass = columns[1]

    

    if pd.isnull(Age):

        

        if Pclass == 1 :

            return 37.0

        

        elif  Pclass == 2 :

            return  29.0

        

        else :

            return 24.0

        

    else:

        return Age
train['Age'] = train[['Age','Pclass']].apply(fill_age,axis=1)
print("The mean after is ", train.Age.mean())
g = sns.catplot(x='Pclass',y='Survived',hue='Sex',data=train,kind='bar',palette= 'RdBu');

g.set_ylabels("Probability of Survival")

g.set_xlabels("Class Number")

g.fig.suptitle("Probability of Survival for Classes");
g = sns.catplot(x='Embarked',y='Survived',hue='Sex',data=train,kind='bar',palette= 'RdBu');

g.set_ylabels("Probability of Survival")

g.set_xlabels("Embarkment Location")

g.fig.suptitle("Probability of Survival by Embarkment Location");
g = sns.catplot(x='SibSp',y='Survived',hue='Sex',data=train,kind='bar',palette= 'RdBu');

g.set_ylabels("Probability of Survival")

g.set_xlabels("# of Siblings")

g.fig.suptitle("Probability of Survival by # of Siblings");
g = sns.catplot(x='Parch',y='Survived',hue='Sex',data=train,kind='bar',palette= 'RdBu');

g.set_ylabels("Probability of Survival")

g.set_xlabels("Parch")

g.fig.suptitle("Probability of Survival # of parents");
plt.figure(figsize=(15,8))

train[train['Survived']==1]['Fare'].hist(alpha=0.5,bins=40,label='1',color='blue')

train[train['Survived']==0]['Fare'].hist(alpha=0.5,bins=40,label='0',color='red')

plt.legend();
train['Fare'].hist(alpha=0.5,bins=40,label='0',color='red');
names = train.Name.str.split(',')

names2 = []

for i in range(0,891):

    names2.append(names[i][1].split('.')[0])
namedummies = pd.get_dummies(names2,drop_first=True)
train = pd.concat([train,namedummies],axis=1)
sex = pd.get_dummies(train['Sex'],drop_first=True)

emb = pd.get_dummies(train['Embarked'],drop_first=True)
train.drop(['Sex','Embarked','Name'],inplace=True, axis=1)
train = pd.concat([train,sex,emb],axis=1);
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train.drop(['Fare','Survived'],axis=1), train['Survived'], test_size=0.33, random_state=42)
logmodel = LogisticRegression()

logmodel.fit(X_train,y_train);
pred = logmodel.predict(X_test)
from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score
test = pd.read_csv('../input/test.csv')

temp_test = test
test.head()
sns.heatmap(test.isnull())
pass_id = test.PassengerId

test.drop(['Cabin','PassengerId','Ticket'],inplace=True, axis=1)
print("The mean before is ", test.Age.mean())

print(test.groupby('Pclass').mean()['Age'])
def age_fill(columns):

    Age = columns[0]

    Pclass = columns[1]

    

    if pd.isnull(Age):

        

        if Pclass == 1 :

            return 40.69

        

        elif  Pclass == 2 :

            return 28.83

        

        else :

            return 24.39

        

    else:

        return Age
test['Age'] = test[['Age','Pclass']].apply(age_fill,axis=1)
namesx = test.Name.str.split(',')

namesx2 = []

for i in range(0,test.shape[0]):

    namesx2.append(namesx[i][1].split('.')[0])
namexdummies = pd.get_dummies(namesx2,drop_first=True)
test = pd.concat([test,namexdummies],axis=1)
sex1 = pd.get_dummies(test['Sex'],drop_first=True)

emb1 = pd.get_dummies(test['Embarked'],drop_first=True)

test.drop(['Sex','Embarked','Name'],inplace=True, axis=1)

test = pd.concat([test,sex1,emb1],axis=1);
test.head()
train.head()
test.fillna(test['Fare'].median(),inplace=True)
new_train = train.drop([' Jonkheer',' the Countess',' Mme',' Mlle',' Major',' Lady',' Col', ' Don', ' Sir'],axis=1)

new_test = test.drop([' Dona'],axis=1) # Recall dropping fare gave us better accuracy.
X_train1, X_test1, y_train1, y_test1 = train_test_split(new_train.drop(['Survived'],axis=1), new_train['Survived'], test_size=0.33, random_state=42)
new_logmodel = LogisticRegression()

new_logmodel.fit(X_train1,y_train1)
prediction = new_logmodel.predict(new_test)
sns.heatmap(temp.corr(),annot=True);
output = pd.DataFrame({ 'PassengerId' : pass_id, 'Survived': prediction })
output.to_csv('titanic-predictions.csv', index = False)
from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix
print(classification_report(y_test1,pred))
score = logmodel.score(X_test, y_test)

print("Accuracy for train.csv is ",score*100,"%")
score1 = new_logmodel.score(X_test1, y_test1)

print("Accuracy for test.csv is ",score1*100,"%")