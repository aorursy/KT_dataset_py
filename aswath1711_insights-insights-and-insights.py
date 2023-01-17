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


import numpy as np

import pandas as pd

import re as re

from matplotlib import pyplot as plt

from matplotlib import style

import seaborn as sns





from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier #KNN
train_df=pd.read_csv("../input/train.csv")

test_df=pd.read_csv("../input/test.csv")

train_df.info()

train_df.describe()

train_df.head()
test_df.head()
train_df.columns.values
total=train_df.isnull().sum().sort_values()

percent_1=train_df.isnull().sum()/train_df.isnull().count()*100

percent_2 = (round(percent_1, 1)).sort_values()

missing_data=pd.concat([total,percent_2],axis=1,keys=['total','percent'])

print(missing_data)

total1=test_df.isnull().sum().sort_values()

percent1_1=test_df.isnull().sum()/test_df.isnull().count()*100

percent1_2 = (round(percent1_1, 1)).sort_values()

missing_data1=pd.concat([total1,percent1_2],axis=1,keys=['total','percent'])

print(missing_data1)
train_df[train_df['Embarked'].isnull()]
train_df["Embarked"] = train_df["Embarked"].fillna('S')

train_df[train_df['Embarked'].isnull()]

survived="Survived"

not_survived="Not_survived"

fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))

women = train_df[train_df['Sex']=='female']

men = train_df[train_df['Sex']=='male']

ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[0], kde =False)

ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[0], kde =False)

ax.legend()

ax.set_title("Women")

ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[1], kde = False)

ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[1], kde = False)

ax.legend()

_ = ax.set_title('Men')

train_df = train_df.drop(['PassengerId'], axis=1)

train_df.head()
train_df.hist(bins=10,figsize=(9,7),grid=False);
train_df.Embarked.value_counts().plot(kind='bar', alpha=0.55)

plt.title("Passengers per boarding location");
train_df.Age[train_df.Pclass == 1].plot(kind='kde')    

train_df.Age[train_df.Pclass == 2].plot(kind='kde')

train_df.Age[train_df.Pclass == 3].plot(kind='kde')

 # plots an axis lable

plt.xlabel("Age")    

plt.title("Age Distribution within classes")

# sets our legend for our graph.

plt.legend(('1st Class', '2nd Class','3rd Class'),loc='best') ;

corr=train_df.corr()#["Survived"]

plt.figure(figsize=(10, 10))



sns.heatmap(corr, vmax=.8, linewidths=0.01,

            square=True,annot=True,cmap='YlGnBu',linecolor="white")

plt.title('Correlation between features');
#train_df["FamilySize"] = train_df["SibSp"] + train_df["Parch"]+1

#test_df["FamilySize"]=test_df["SibSp"] + test_df["Parch"] +1

#print(train_df["FamilySize"].value_counts())
train_df = train_df.drop(['Name','Ticket', 'Cabin'], axis=1)

test_df = test_df.drop(['Name','Ticket', 'Cabin'], axis=1)

train_df.head()


sex_mapping = {"male": 0, "female": 1}

train_df['Sex'] = train_df['Sex'].map(sex_mapping)

test_df['Sex'] = test_df['Sex'].map(sex_mapping)



train_df.head()
embarked_mapping = {"S": 1, "C": 2, "Q": 3}

train_df['Embarked'] = train_df['Embarked'].map(embarked_mapping)

test_df['Embarked'] = test_df['Embarked'].map(embarked_mapping)



train_df.head()
guess_ages = np.zeros((2,3))

guess_ages

combine = [train_df, test_df]

for dataset in combine:

    for i in range(0, 2):

        for j in range(0, 3):

            guess_df = dataset[(dataset['Sex'] == i) & \

                                  (dataset['Pclass'] == j+1)]['Age'].dropna()



            # age_mean = guess_df.mean()

            # age_std = guess_df.std()

            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)



            age_guess = guess_df.median()



            # Convert random age float to nearest .5 age

            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

            

    for i in range(0, 2):

        for j in range(0, 3):

            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\

                    'Age'] = guess_ages[i,j]



    dataset['Age'] = dataset['Age'].astype(int)



train_df.head()
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)

train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)

for dataset in combine:    

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age']

train_df.head()

train_df = train_df.drop(['AgeBand'], axis=1)
data = [train_df, test_df]



for dataset in data:

    dataset['Fare'] = dataset['Fare'].fillna(0)

    dataset['Fare'] = dataset['Fare'].astype(int)
data = [train_df, test_df]



for dataset in data:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare']   = 3

    dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare']   = 4

    dataset.loc[ dataset['Fare'] > 250, 'Fare'] = 5

    dataset['Fare'] = dataset['Fare'].astype(int)

train_df['Fare'].value_counts()
train_df.head()

X_train = train_df.drop("Survived", axis=1)

Y_train = train_df["Survived"]

X_test  = test_df.drop("PassengerId", axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape
logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

acc_log
decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

acc_decision_tree
"""knn = KNeighborsClassifier(algorithm='auto', leaf_size=26, metric='minkowski', 

                           metric_params=None, n_jobs=1, n_neighbors=6, p=2, 

                           weights='uniform')

knn.fit(X_train, Y_train)

y_pred = knn.predict(X_test)

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

acc_knn"""
#gradient boosting algorithm

from sklearn.ensemble import GradientBoostingClassifier

gradient_boost = GradientBoostingClassifier()

gradient_boost.fit(X_train, Y_train)

y_pred = gradient_boost.predict(X_test)

#extra trees classifier

from sklearn.ensemble import ExtraTreesClassifier

ExtraTreesClassifier = ExtraTreesClassifier()

ExtraTreesClassifier.fit(X_train, Y_train)

y_pred = ExtraTreesClassifier.predict(X_test)

submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })
submission.to_csv('submission', index=False)