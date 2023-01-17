# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas import Series,DataFrame

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB
trainData = pd.read_csv("../input/train.csv", )

testData = pd.read_csv("../input/test.csv")

trainData.head()

trainData = trainData.drop(['PassengerId', 'Name', 'Ticket'], axis = 1)

testData = testData.drop(['PassengerId', 'Name'], axis = 1)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
trainData['Embarked'] = trainData['Embarked'].fillna('S')



sns.factorplot('Embarked','Survived', data=trainData,size=4,aspect=3)



fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5))



sns.countplot(x='Survived', hue="Embarked", data=trainData, order=[1,0], ax=axis2)



embark_perc = trainData[["Embarked", "Survived"]].groupby(['Embarked'],as_index=False).mean()

sns.barplot('Embarked', 'Survived', data=embark_perc)



embark_dummies_titanic  = pd.get_dummies(trainData['Embarked'])

embark_dummies_titanic.drop(['S'], axis=1, inplace=True)



embark_dummies_test  = pd.get_dummies(testData['Embarked'])

embark_dummies_test.drop(['S'], axis=1, inplace=True)



trainData = trainData.join(embark_dummies_titanic)

testData = testData.join(embark_dummies_test)



trainData.drop(['Embarked'], axis=1,inplace=True)

testData.drop(['Embarked'], axis=1,inplace=True)
# Fare



# only for test_df, since there is a missing "Fare" values

testData["Fare"].fillna(testData["Fare"].median(), inplace=True)



# convert from float to int

trainData['Fare'] = trainData['Fare'].astype(int)

testData['Fare']    = testData['Fare'].astype(int)



# get fare for survived & didn't survive passengers 

fare_not_survived = trainData["Fare"][trainData["Survived"] == 0]

fare_survived     = trainData["Fare"][trainData["Survived"] == 1]



# get average and std for fare of survived/not survived passengers

avgerage_fare = DataFrame([fare_not_survived.mean(), fare_survived.mean()])

std_fare      = DataFrame([fare_not_survived.std(), fare_survived.std()])



# plot

trainData['Fare'].plot(kind='hist', figsize=(15,3),bins=100, xlim=(0,50))



avgerage_fare.index.names = std_fare.index.names = ["Survived"]

avgerage_fare.plot(yerr=std_fare,kind='bar',legend=False)
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))

axis1.set_title('Original Age values - Titanic')

axis2.set_title('New Age values - Titanic')



# axis3.set_title('Original Age values - Test')

# axis4.set_title('New Age values - Test')



# get average, std, and number of NaN values in titanic_df

average_age_titanic   = trainData["Age"].mean()

std_age_titanic       = trainData["Age"].std()

count_nan_age_titanic = trainData["Age"].isnull().sum()



# get average, std, and number of NaN values in test_df

average_age_test   = testData["Age"].mean()

std_age_test       = testData["Age"].std()

count_nan_age_test = testData["Age"].isnull().sum()



# generate random numbers between (mean - std) & (mean + std)

rand_1 = np.random.randint(average_age_titanic - std_age_titanic, average_age_titanic + std_age_titanic, size = count_nan_age_titanic)

rand_2 = np.random.randint(average_age_test - std_age_test, average_age_test + std_age_test, size = count_nan_age_test)



# plot original Age values

# NOTE: drop all null values, and convert to int

trainData['Age'].dropna().astype(int).hist(bins=70, ax=axis1)

# test_df['Age'].dropna().astype(int).hist(bins=70, ax=axis1)



# fill NaN values in Age column with random values generated

trainData["Age"][np.isnan(trainData["Age"])] = rand_1

testData["Age"][np.isnan(testData["Age"])] = rand_2



# convert from float to int

trainData['Age'] = trainData['Age'].astype(int)

testData['Age']    = testData['Age'].astype(int)

        

# plot new Age Values

trainData['Age'].hist(bins=70, ax=axis2)
facet = sns.FacetGrid(trainData, hue="Survived",aspect=4)

facet.map(sns.kdeplot,'Age',shade= True)

facet.set(xlim=(0, trainData['Age'].max()))

facet.add_legend()



fig, axis1 = plt.subplots(1,1,figsize=(18,4))

average_age = trainData[["Age", "Survived"]].groupby(['Age'],as_index=False).mean()

sns.barplot(x='Age', y='Survived', data=average_age)
trainData.drop("Cabin",axis=1,inplace=True)

testData.drop("Cabin",axis=1,inplace=True)
def get_person(passenger):

    age,sex = passenger

    return 'child' if age < 16 else sex

    

trainData['Person'] = trainData[['Age','Sex']].apply(get_person,axis=1)

testData['Person']    = testData[['Age','Sex']].apply(get_person,axis=1)



trainData = trainData.drop('Sex', axis=1)

testData = testData.drop('Sex', axis=1)



person_dummies_titanic  = pd.get_dummies(trainData['Person'])

person_dummies_titanic.columns = ['Child','Female','Male']

person_dummies_titanic.drop(['Male'], axis=1, inplace=True)



person_dummies_test  = pd.get_dummies(testData['Person'])

person_dummies_test.columns = ['Child','Female','Male']

person_dummies_test.drop(['Male'], axis=1, inplace=True)



trainData = trainData.join(person_dummies_titanic)

testData    = testData.join(person_dummies_test)



trainData.head()



fig, (axis1,axis2) = plt.subplots(1,2,figsize=(10,5))

sns.countplot('Person', data=trainData, ax=axis1)

person_perc = trainData[["Person", "Survived"]].groupby(['Person'],as_index=False).mean()

sns.barplot(x='Person', y='Survived', data=person_perc, ax=axis2, order=['male','female','child'])



trainData.drop(['Person'],axis=1,inplace=True)

testData.drop(['Person'],axis=1,inplace=True)
# Pclass



# sns.factorplot('Pclass',data=titanic_df,kind='count',order=[1,2,3])

sns.factorplot('Pclass','Survived',order=[1,2,3], data=trainData,size=5)



# create dummy variables for Pclass column, & drop 3rd class as it has the lowest average of survived passengers

pclass_dummies_titanic  = pd.get_dummies(trainData['Pclass'])

pclass_dummies_titanic.columns = ['Class_1','Class_2','Class_3']

pclass_dummies_titanic.drop(['Class_3'], axis=1, inplace=True)



pclass_dummies_test  = pd.get_dummies(testData['Pclass'])

pclass_dummies_test.columns = ['Class_1','Class_2','Class_3']

pclass_dummies_test.drop(['Class_3'], axis=1, inplace=True)



trainData.drop(['Pclass'],axis=1,inplace=True)

testData.drop(['Pclass'],axis=1,inplace=True)



trainData = trainData.join(pclass_dummies_titanic)

testData    = testData.join(pclass_dummies_test)
trainData.head()
X_train = trainData.drop("Survived",axis=1)

Y_train = trainData["Survived"]

X_test  = testData.drop("Ticket", axis = 1).copy()
X_train
random_forest = RandomForestClassifier(n_estimators=100)



random_forest.fit(X_train, Y_train)



Y_pred = random_forest.predict(X_test)



random_forest.score(X_train, Y_train)
testData
testData = pd.read_csv("../input/test.csv")

submission = pd.DataFrame({

        "PassengerId": testData["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('titanic.csv', index=False)