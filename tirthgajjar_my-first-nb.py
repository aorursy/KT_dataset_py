# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import VotingClassifier

from subprocess import check_output
train_df = pd.read_csv("../input/train.csv")

test_df  = pd.read_csv("../input/test.csv")



train_df.head()
#Count missing values for each column

train_df.info()

test_df.info()
train_df['Embarked'].value_counts().idxmax()
#Add missing values for train and test datasets

#Count missing values



null_age_count_train = train_df["Age"].isnull().sum()

print(null_age_count_train)



null_age_count_test = test_df["Age"].isnull().sum()

print(null_age_count_test)



avg_age_train = train_df["Age"].mean()

std_age_train = train_df["Age"].std()

replacement_random_vals_train = np.random.randint(avg_age_train-std_age_train,avg_age_train+std_age_train,size=null_age_count_train)

print(replacement_random_vals_train)

avg_age_test = test_df['Age'].mean()

std_age_test = test_df['Age'].std()

replacement_random_vals_test = np.random.randint(avg_age_test-std_age_test,avg_age_test+std_age_test,size=null_age_count_test)

print(replacement_random_vals_test)



train_df["Age"][np.isnan(train_df["Age"])] = replacement_random_vals_train

test_df["Age"][np.isnan(test_df["Age"])] = replacement_random_vals_test



train_df.info()

test_df.info()
train_df['Age'] = train_df['Age'].astype(int)

test_df['Age']  = test_df['Age'].astype(int)



test_df["Fare"].fillna(test_df["Fare"].median(), inplace=True)

train_df['Fare'] = train_df['Fare'].astype(int)

test_df['Fare']  = test_df['Fare'].astype(int)
sns.factorplot('Sex','Survived', data=train_df,size=5)




def get_person(passenger):

    age,sex = passenger

    return 'child' if age < 16 else sex

    

train_df['Person'] = train_df[['Age','Sex']].apply(get_person,axis=1)

test_df['Person']  = test_df[['Age','Sex']].apply(get_person,axis=1)



train_df.drop(['Sex'],axis=1,inplace=True)

test_df.drop(['Sex'],axis=1,inplace=True)



person_dummies_train  = pd.get_dummies(train_df['Person'])

person_dummies_train.columns = ['Child','Female','Male']

person_dummies_train.drop(['Male'], axis=1, inplace=True)



person_dummies_test  = pd.get_dummies(test_df['Person'])

person_dummies_test.columns = ['Child','Female','Male']

person_dummies_test.drop(['Male'], axis=1, inplace=True)



train_df = train_df.join(person_dummies_train)

test_df  = test_df.join(person_dummies_test)

fig, (axis1,axis2) = plt.subplots(1,2,figsize=(10,5))

sns.countplot(x='Person', data=train_df, ax=axis1)



person_perc = train_df[["Person", "Survived"]].groupby(['Person'],as_index=False).mean()

sns.barplot(x='Person', y='Survived', data=person_perc,  ax=axis2)



train_df.drop(['Person'],axis=1,inplace=True)

test_df.drop(['Person'],axis=1,inplace=True)
sns.factorplot('Pclass','Survived',order=[1,2,3], data=train_df,size=5)
pclass_dummies_train  = pd.get_dummies(train_df['Pclass'])

pclass_dummies_train.columns = ['Class_1','Class_2','Class_3']

pclass_dummies_train.drop(['Class_3'], axis=1, inplace=True)



pclass_dummies_test  = pd.get_dummies(test_df['Pclass'])

pclass_dummies_test.columns = ['Class_1','Class_2','Class_3']

pclass_dummies_test.drop(['Class_3'], axis=1, inplace=True)



train_df.drop(['Pclass'],axis=1,inplace=True)

test_df.drop(['Pclass'],axis=1,inplace=True)



train_df = train_df.join(pclass_dummies_train)

test_df    = test_df.join(pclass_dummies_test)
train_df.head()
train_df = train_df.drop(['PassengerId','Name','Ticket'], axis=1)

test_df    = test_df.drop(['Name','Ticket'], axis=1)



train_df['Family'] =  train_df["Parch"] + train_df["SibSp"]

train_df['Family'].loc[train_df['Family'] > 0] = 1

train_df['Family'].loc[train_df['Family'] == 0] = 0



test_df['Family'] =  test_df["Parch"] + test_df["SibSp"]

test_df['Family'].loc[test_df['Family'] > 0] = 1

test_df['Family'].loc[test_df['Family'] == 0] = 0



# drop Parch & SibSp

train_df = train_df.drop(['SibSp','Parch'], axis=1)

test_df    = test_df.drop(['SibSp','Parch'], axis=1)



train_df = train_df.drop(['Cabin'], axis=1)

test_df = test_df.drop(['Cabin'], axis=1)

train_df = train_df.drop(['Embarked'], axis=1)

test_df = test_df.drop(['Embarked'], axis=1)

train_df.head()
X_train = train_df.drop("Survived",axis=1)

Y_train = train_df["Survived"]

X_test  = test_df.drop("PassengerId",axis=1).copy()



random_forest = RandomForestClassifier(n_estimators=10)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)



lr = LogisticRegression(random_state=7)

lr.fit(X_train,Y_train)

knn = KNeighborsClassifier()

knn.fit(X_train,Y_train)
eclf1 = VotingClassifier(estimators=[

        ('lr', lr), ('rf', random_forest), ('knn', knn)], voting='soft', weights=[1, 2, 1])

eclf1 = eclf1.fit(X_train,Y_train)

results = eclf1.predict(X_test)

output = pd.DataFrame({'PassengerId': test_df['PassengerId'], "Survived": results})

output.to_csv('prediction.csv', index=False)
print(check_output(["ls"]).decode("utf8"))