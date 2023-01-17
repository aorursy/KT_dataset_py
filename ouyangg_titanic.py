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



from sklearn.linear_model import Perceptron,LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn import svm

from sklearn.model_selection import train_test_split
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

train_data.head()
train_data.info()
train_data.isnull().mean().sort_values(ascending=False)
sns.countplot('Embarked',data=train_data)
train_data[['Embarked','Survived']].groupby(['Embarked'],as_index=False).mean().sort_values(by='Embarked',ascending=False)
sns.barplot('Embarked','Survived',data=train_data)
train_data[['Sex','Survived']].groupby(['Sex'],as_index=False).mean().sort_values(by='Sex',ascending=False)
train_data[['SibSp','Survived']].groupby(['SibSp'],as_index=False).mean().sort_values(by='SibSp',ascending=False)
sns.barplot('SibSp','Survived',data=train_data)
sns.barplot('Parch','Survived',data=train_data)
grid1 = sns.FacetGrid(train_data,row='Embarked',col='Survived',size=2.2,aspect=1.6)

grid1.map(plt.hist,'Age',alpha=0.5,bins=20)

grid1.add_legend()
sns.barplot('Survived','Fare',data=train_data)
plt.figure(figsize=(8,8),dpi=80)

sns.heatmap(train_data.drop('PassengerId',axis=1).corr(),annot=True,square=True,cmap = 'YlGnBu',linewidth=2,linecolor='black',annot_kws={'size':8})
sns.barplot('Pclass','Survived',data=train_data)
grid2 = sns.FacetGrid(train_data,row='Sex',col='Survived',size=2.2,aspect=1.8)

grid2.map(plt.hist,'Age',alpha=0.5,bins=25)
train_data.head()
combine = [train_data,test_data]

print('Before:',combine[0].shape,combine[1].shape)

for i,dataset in enumerate(combine):

    dataset = dataset.drop(['Ticket','Cabin'],axis=1)

    combine[i] = dataset

print('After:',combine[0].shape,combine[1].shape)
for i,dataset in enumerate(combine):

    dataset['Sex']=dataset['Sex'].map({'male':0,'female':1})

combine[0].head()
for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].fillna('S')

combine[0].isnull().mean().sort_values(ascending=False)

combine[1].isnull().mean().sort_values(ascending=False)
median = combine[1][['Fare','Pclass','Sex']].groupby(['Pclass','Sex'],as_index=False).median()

median
combine[1].loc[(combine[1]['Fare'].isnull())&(combine[1]['Pclass']==1)&(combine[1]['Sex']==0),'Fare'] = 51.86250

combine[1].loc[(combine[1]['Fare'].isnull())&(combine[1]['Pclass']==1)&(combine[1]['Sex']==1),'Fare'] = 79.02500

combine[1].loc[(combine[1]['Fare'].isnull())&(combine[1]['Pclass']==2)&(combine[1]['Sex']==0),'Fare'] = 13

combine[1].loc[(combine[1]['Fare'].isnull())&(combine[1]['Pclass']==2)&(combine[1]['Sex']==1),'Fare']= 26.00000

combine[1].loc[(combine[1]['Fare'].isnull())&(combine[1]['Pclass']==3)&(combine[1]['Sex']==0),'Fare'] = 7.89580

combine[1].loc[(combine[1]['Fare'].isnull())&(combine[1]['Pclass']==3)&(combine[1]['Sex']==1),'Fare'] = 8.08125
combine[0].isnull().mean().sort_values(ascending=False)
combine[0].isnull().mean().sort_values(ascending=False)
mean = pd.concat([combine[0],combine[1]])[['Age','Pclass']].groupby(['Pclass'],as_index=False).mean()

mean
for dataset in combine:

    dataset.loc[(dataset['Age'].isnull())&(dataset['Pclass']==1),'Age'] = 39.159930

    dataset.loc[(dataset['Age'].isnull())&(dataset['Pclass']==2),'Age'] = 29.506705

    dataset.loc[(dataset['Age'].isnull())&(dataset['Pclass']==3),'Age'] = 24.816367

combine[0].isnull().mean()
Embarked = {'C':0,'S':2,'Q':1}

for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].map(Embarked)

combine[0].head()
combine[0]['Title'] = combine[0].Name.str.extract('([A-Za-z]+)\.')
combine[0]['Title'].value_counts()
Title = {'Mr':0,'Miss':0,'Mrs':0,'Master':0,'Dr':0,'Ms':1,'Rev':1,'Major':1,'Col':1,\

          'Mlle':1,'Countess':1,'Capt':1,'Don':1,'Sir':1,'Lady':1,'Sir':1,'Lady':1,'Jonkheer':1,'Mme':1}

combine[0].Title = combine[0].Title.map(Title)

combine[0].head()
combine[0] = combine[0].drop('Name',axis=1)

combine[0].head()
combine[1]['Title'] = combine[1].Name.str.extract('([A-Za-z]+)\.')
combine[1].Title.value_counts()
Title2 = {'Mr':0,'Miss':0,'Mrs':0,'Master':0,'Rev':1,'Col':1,'Dr':1,'Dona':1,'Ms':1}

combine[1]['Title'] = combine[1]['Title'].map(Title2)

combine[1].head()
pd.cut(pd.concat([combine[0],combine[1]])['Age'],5).value_counts()
for dataset in combine:

    dataset.loc[(dataset['Age']<16),'Age'] = 0

    dataset.loc[(16<=dataset['Age'])&(dataset['Age']<32),'Age'] = 1

    dataset.loc[(32<=dataset['Age'])&(dataset['Age']<48),'Age'] = 2

    dataset.loc[(48<=dataset['Age'])&(dataset['Age']<64),'Age'] = 3

    dataset.loc[(64<=dataset['Age']),'Age'] = 4

combine[0].head()
combine[0]['Age'] = combine[0]['Age'].astype(int)

combine[1]['Age'] = combine[1]['Age'].astype(int)
pd.cut(pd.concat([combine[0],combine[1]])['Fare'],4).value_counts()
for dataset in combine:

    dataset.loc[(dataset['Fare']<128),'Fare'] = 0

    dataset.loc[(128<=dataset['Fare'])&(dataset['Fare']<256),'Fare'] = 1

    dataset.loc[(256<=dataset['Fare'])&(dataset['Fare']<384),'Fare'] = 2

    dataset.loc[(384<=dataset['Fare']),'Fare'] = 3

combine[0]['Fare'] = combine[0]['Fare'].astype(int)

combine[1]['Fare'] = combine[1]['Fare'].astype(int)

combine[0].head()
print(combine[0].shape,combine[1].shape)
for dataset in combine:

    dataset['Family_size'] = dataset['SibSp'] + dataset['Parch'] + 1 #need to count myself

print(combine[0].head())
combine[0]['Family_size'].value_counts()
combine[0]['Single'] = 0

combine[0]['SmallF'] = 0

combine[0]['MidF'] = 0

combine[0]['LargeF'] = 0

combine[1]['Single'] = 0

combine[1]['SmallF'] = 0

combine[1]['MidF'] = 0

combine[1]['LargeF'] = 0

combine[0].head()
for dataset in combine:

    dataset.loc[(dataset['Family_size']==1),'Single']=1

    dataset.loc[(dataset['Family_size']>1)&(dataset['Family_size']<=3),'SmallF']=1

    dataset.loc[(dataset['Family_size']>3)&(dataset['Family_size']<=6),'MidF']=1

    dataset.loc[dataset['Family_size']>=6,'LargeF'] = 1 

combine[0].head()
train_set = combine[0]

test_set = combine[1]

X = train_set.drop(['PassengerId','Survived','SibSp','Parch','Family_size'],axis=1)

Y = train_set['Survived']

X_train,X_val,Y_train,Y_val = train_test_split(X,Y,test_size=0.3,shuffle=True)

print(X_train.shape,X_val.shape,Y_train.shape,Y_val.shape)
LR = LogisticRegression()

LR.fit(X_train,Y_train)

LR_score = LR.score(X_val,Y_val)

LR_score
LR.score(X_train,Y_train)
SVM_score_list =[]

C_list = [1,10,50,100,1000,2000]

gamma_list = [0.0001,0.001,0.002,0.004,0.01,0.1]

HyperParameter = {}

best_score = 0.0

for C in C_list:

    for gamma in gamma_list:

        SVM = svm.SVC(C=C,gamma=gamma)

        SVM.fit(X_train,Y_train)

        SVM_score=SVM.score(X_val,Y_val)

        SVM_score_list.append(SVM_score)

        if SVM_score>best_score:

            best_score = SVM_score

            HyperParameter['C'] = C

            HyperParameter['gamma'] = gamma

            HyperParameter['Score'] = best_score

plt.plot(np.arange(len(SVM_score_list)),SVM_score_list)

plt.show()
HyperParameter
SVM = svm.SVC(C=1000,gamma=0.01)

SVM.fit(X_train,Y_train)

SVM.score(X_val,Y_val)
SVM.score(X_train,Y_train)
RFC = RandomForestClassifier(n_estimators=100,

                             criterion='gini',max_depth=4

                             )

RFC.fit(X_train,Y_train)

RFC.score(X_val,Y_val)
RFC.score(X_train,Y_train)
DT = DecisionTreeClassifier(criterion='gini')

DT.fit(X_train,Y_train)

DT.score(X_val,Y_val)
X_test = test_set.drop(['PassengerId','Name','SibSp','Parch','Family_size'],axis=1)

X_test.head()
Y_id = test_set.PassengerId

prediction = SVM.predict(X_test)

print(prediction.shape)
result = pd.DataFrame({'PassengerId':Y_id,'Survived':prediction})

print(result.shape)
result.to_csv('./RRResult.csv',index=False)

print('Finished!')