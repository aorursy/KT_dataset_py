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

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')
train.info()
train.describe()
test.info()
# Study Pclass

pd.pivot_table(train, index = 'Pclass', values = ['Survived'])
# Study name

train.Name.head(10)
# Split the name into family and title

train['Title'] = train.Name.apply(lambda x: x.split(',')[1].split('.')[0])

train['Family'] = train.Name.apply(lambda x: x.split(',')[0])



print('Title values')

print(train.Title.value_counts())

print('Family values')

print(train.Family.value_counts())
train = train.drop('Family',axis=1)

pd.pivot_table(train, index = 'Title', values = ['Survived'])
train['Title'] = train.Title.apply(lambda x: x.replace('Mlle','Miss').replace('Mme','Mrs').replace('Col','Others').replace('Major','Others').replace('Don','Others').replace('Lady','Others').replace('Sir','Others').replace('Capt','Others').replace('the Countess','Others').replace('Ms','Others').replace('Jonkheer','Others'))

pd.pivot_table(train, index = 'Title', values = ['Survived'])



# I apply all this to the test dataset

test['Title'] = test.Name.apply(lambda x: x.split(',')[1].split('.')[0])

test['Title'] = test.Title.apply(lambda x: x.replace('Mlle','Miss').replace('Mme','Mrs').replace('Col','Others').replace('Major','Others').replace('Don','Others').replace('Lady','Others').replace('Sir','Others').replace('Capt','Others').replace('the Countess','Others').replace('Ms','Others').replace('Jonkheer','Others'))

test.head()
# I will remove Name from both datasets

train = train.drop('Name',axis = 1)

test = test.drop('Name',axis = 1)
# Study sex

pd.pivot_table(train, index = 'Sex', values = ['Survived'])

# Study Age,SibSp ,Parch and Fare

# Isolate survivors

surv = train.copy()

surv.drop(surv[surv['Survived']==0].index, inplace=True)

num = ['Age','SibSp','Parch','Fare']

for x in num:

    fig = plt.figure(figsize=(15,15))

    ax1 = plt.subplot2grid((3,2),(0,0))

    plt.hist(train[x])

    plt.title(x +' Total')

    ax1 = plt.subplot2grid((3,2),(0,1))

    plt.hist(surv[x])

    plt.title(x + ' Survived')

    plt.show()



print('Total')

print(train[num].describe())

print('Survived')

print(surv[num].describe())
# Check the correlations

num.append('Survived')

print(train[num].corr())

sns.heatmap(train[num].corr())
ageMean = train.groupby('Title')['Age'].median()

meanDr = ageMean[0]

meanMaster = ageMean[1]

meanMiss = ageMean[2]

meanMr = ageMean[3]

meanMrs = ageMean[4]

meanOthers = ageMean[5]

meanRev = ageMean[6]

ageMean

# Fill NA with the mean based on his Title

for x in range(len(train["Age"])):

    if pd.isna(train["Age"][x]):

        title = train["Title"][x]

        title = title.strip() # This is for remove spaces

        if(title == 'Dr'):

            train["Age"][x] = meanDr

           

        elif(title == "Master"):

            train["Age"][x] = meanMaster

           

        elif(title == "Miss"):

            train["Age"][x] = meanMiss



        elif(title =="Mr"):

            train["Age"][x] = meanMr

        elif(title == "Mrs"):

            train["Age"][x] = meanMrs

        elif(title == "Others"):

            train["Age"][x] = meanOthers

        elif(title == "Rev"):

            train["Age"][x] = meanRev

            

for x in range(len(test["Age"])):

    if pd.isna(test["Age"][x]):

        title = test["Title"][x]

        title = title.strip() # This is for remove spaces

        if(title == 'Dr'):

            test["Age"][x] = meanDr

           

        elif(title == "Master"):

            test["Age"][x] = meanMaster

           

        elif(title == "Miss"):

            test["Age"][x] = meanMiss



        elif(title =="Mr"):

            test["Age"][x] = meanMr

        elif(title == "Mrs"):

            test["Age"][x] = meanMrs

        elif(title == "Others"):

            test["Age"][x] = meanOthers

        elif(title == "Rev"):

            test["Age"][x] = meanRev

# Group by age

train['AgeGroup'] = train.Age.apply(lambda x: 'Baby' if x <= 5 else ('Child' if x <= 12 else('Teenager' if x <= 18 else ('Young') if x <= 24 else ('Young Adult' if x <= 35 else ('Adult' if x <= 60 else 'Senior')))) )

print(pd.pivot_table(train, index = 'AgeGroup', values = ['Survived']))

sns.barplot(x="AgeGroup", y="Survived", data=train)

plt.show()
test['AgeGroup'] = test.Age.apply(lambda x: 'Baby' if x <= 5 else ('Child' if x <= 12 else('Teenager' if x <= 18 else ('Young') if x <= 24 else ('Young Adult' if x <= 35 else ('Adult' if x <= 60 else 'Senior')))) )

# Study Ticket

train.Ticket


train['NumericTicket'] = train.Ticket.apply(lambda x: 1 if x.isnumeric() else 0)

pd.pivot_table(train, index = 'NumericTicket', values = ['Survived'])


train = train.drop('NumericTicket',axis=1)

train = train.drop('Ticket',axis=1)

test = test.drop('Ticket',axis=1)
# Study Cabin

train.Cabin


train['has_cabin'] = train.Cabin.apply(lambda x: 0 if pd.isna(x) else 1)

train['CabinLetter'] = train.Cabin.apply(lambda x: str(x)[0])

train.head()
# Check if has_cabin is useful

print(pd.pivot_table(train, index = 'has_cabin', values = ['Survived']))

sns.barplot(x="has_cabin", y="Survived", data=train)

plt.show()
print(pd.pivot_table(train, index = 'CabinLetter', values = ['Survived']))

sns.barplot(x="CabinLetter", y="Survived", data=train)

plt.show()
test['has_cabin'] = test.Cabin.apply(lambda x: 0 if pd.isna(x) else 1)

test['CabinLetter'] = test.Cabin.apply(lambda x: str(x)[0])

#Remove Cabin

train = train.drop('Cabin',axis=1)

test = test.drop('Cabin',axis=1)
# Study embarked and what to do with the NA values

print(pd.pivot_table(train, index = 'Embarked', values = ['Survived']))

sns.barplot(x="Embarked", y="Survived", data=train)

plt.show()

print('Total of passengers for each port')

print(train.Embarked.value_counts())


train = train.fillna({"Embarked": "S"})
train.info()
test.info()
test = test.fillna({"Fare": train.Fare.mean()})

# Study and correct (if it is necessary) the skewness and kurtosis.

print("Train")

print("Skewness:", train['Fare'].skew())

print("Kurtosis: ",train['Fare'].kurt())



plt.hist(train.Fare, bins=10, color='mediumpurple',alpha=0.5)

plt.show()



print("Test")                         

print("Skewness:", test['Fare'].skew())

print("Kurtosis: ",test['Fare'].kurt())



plt.hist(test.Fare, bins=10, color='mediumpurple',alpha=0.5)

plt.show()
train["Fare"] = np.log1p(train["Fare"])

test["Fare"] = np.log1p(test["Fare"])



print("Train")

print("Skewness:", train['Fare'].skew())

print("Kurtosis: ",train['Fare'].kurt())



plt.hist(train.Fare, bins=10, color='mediumpurple',alpha=0.5)

plt.show()



print("Test")                         

print("Skewness:", test['Fare'].skew())

print("Kurtosis: ",test['Fare'].kurt())



plt.hist(test.Fare, bins=10, color='mediumpurple',alpha=0.5)

plt.show()

train = pd.get_dummies(train)

test = pd.get_dummies(test)
# Scale data 

from sklearn.preprocessing import StandardScaler

scale = StandardScaler()



train[['Age','SibSp','Parch','Fare']]= scale.fit_transform(train[['Age','SibSp','Parch','Fare']])

test[['Age','SibSp','Parch','Fare']]= scale.fit_transform(test[['Age','SibSp','Parch','Fare']])
#Split to train

from sklearn.model_selection import train_test_split



pred = train.drop(['Survived', 'PassengerId'], axis=1)

target = train["Survived"]

X_train, X_valid, y_train, y_valid = train_test_split(pred, target, test_size = 0.20, random_state = 0)

# Logistic Regression

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score



logreg = LogisticRegression()

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_valid)

acc_logreg = round(accuracy_score(y_pred, y_valid) * 100, 2)

print(acc_logreg)
# Random Forest

from sklearn.ensemble import RandomForestClassifier



randomforest = RandomForestClassifier()

randomforest.fit(X_train, y_train)

y_pred = randomforest.predict(X_valid)

acc_randomforest = round(accuracy_score(y_pred, y_valid) * 100, 2)

print(acc_randomforest)


ids = test['PassengerId']

predictions = randomforest.predict(test.drop('PassengerId', axis=1))





output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

output.to_csv('submission.csv', index=False)