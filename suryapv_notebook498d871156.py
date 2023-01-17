# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train_file = pd.read_csv("../input/train.csv")

test_file = pd.read_csv("../input/test.csv")

train_file.head(2)
test_file.head(2)

train_df = train_file.drop(['PassengerId','Ticket','Name'],axis = 1)

test_df = test_file.drop(['Ticket','Name'],axis = 1)




train_df['Embarked'] = train_df['Embarked'].fillna('S')

percentage_survived = train_df[['Embarked','Survived']].groupby(['Embarked'],as_index=False).mean()

print(percentage_survived)

sns.barplot(x='Embarked', y='Survived', data = percentage_survived,order=['S','C','Q'])
train_df = train_df.drop('Cabin',axis = 1)

test_df = test_df.drop('Cabin',axis = 1)
#Age Column

count_null_train_df = train_df['Age'].isnull().sum()

count_null_test_df = test_df['Age'].isnull().sum()

mean_train_df = train_df['Age'].mean()

std_train_df = train_df['Age'].std()

print('mean and std deviation is--', mean_train_df, std_train_df)



mean_test_df = test_df['Age'].mean()

std_test_df = test_df['Age'].std()

#generate random numbers

rand1 = np.random.randint(mean_train_df - std_train_df,mean_train_df + std_train_df,size = count_null_train_df)

rand2 = np.random.randint(mean_test_df - std_test_df,mean_test_df + std_test_df,size = count_null_test_df)

train_df["Age"][np.isnan(train_df["Age"])] = rand1

test_df["Age"][np.isnan(test_df["Age"])] = rand2



#train_df['Age'] = train_df['Age'].fillna(rand1)

#test_df['Age'] = test_df['Age'].fillna(rand2)     



train_df['Age'].hist(bins=70)
#Sex Vs. Survived

percentage_survived_sex = train_df[['Sex','Survived']].groupby(['Sex'],as_index=False).mean()

sns.barplot(x='Sex', y='Survived', data = percentage_survived_sex,order=['male','female'])



person_dummies_titanic  = pd.get_dummies(train_df['Sex'])

person_dummies_titanic.columns = ['female','male']

print(person_dummies_titanic)
#Passenger Class Vs. Survived

percentage_survived_class= train_df[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean()

sns.barplot(x='Pclass', y='Survived', data = percentage_survived_class,order=[1,2,3])
#Age



# get average, std, and number of NaN values in train_df

average_age_titanic   = train_df["Age"].mean()

std_age_titanic       = train_df["Age"].std()

count_nan_age_titanic = train_df["Age"].isnull().sum()



# get average, std, and number of NaN values in test_df

average_age_test   = test_df["Age"].mean()

std_age_test       = test_df["Age"].std()

count_nan_age_test = test_df["Age"].isnull().sum()



# generate random numbers between (mean - std) & (mean + std)

rand_1 = np.random.randint(average_age_titanic - std_age_titanic, average_age_titanic + std_age_titanic, size = count_nan_age_titanic)

rand_2 = np.random.randint(average_age_test - std_age_test, average_age_test + std_age_test, size = count_nan_age_test)





train_df["Age"][np.isnan(train_df["Age"])] = rand_1

test_df["Age"][np.isnan(test_df["Age"])] = rand_2

train_df['Age'] = train_df['Age'].astype(int)

test_df['Age']    = test_df['Age'].astype(int)

        

# plot new Age Values

train_df['Age'].hist(bins=70)
#Sex

percentage_survived = train_df[['Sex','Survived']].groupby(['Sex'],as_index=False).mean()

sns.barplot(x='Sex', y='Survived', data = percentage_survived,order=['male','female'])
#Pclass

percentage_survived = train_df[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean()

sns.barplot(x='Pclass', y='Survived', data = percentage_survived,order=[1, 2, 3])
#Sex Column Conversion to Integers

sex_dummies_train = pd.get_dummies(train_df['Sex'])

sex_dummies_test = pd.get_dummies(test_df['Sex'])

train_df = train_df.drop('Sex',axis = 1)

test_df = test_df.drop('Sex',axis = 1)

print(train_df.head(2))

train_df = train_df.join(sex_dummies_train)

print(train_df.head(2))
test_df = test_df.join(sex_dummies_test)
# Fare

from pandas import Series,DataFrame

# only for test_df, since there is a missing "Fare" values

test_df["Fare"].fillna(test_df["Fare"].median(), inplace=True)

train_df['Fare'] = train_df['Fare'].astype(int)

test_df['Fare']    = test_df['Fare'].astype(int)





# get fare for survived & didn't survive passengers 

fare_not_survived = train_df["Fare"][train_df["Survived"] == 0]

fare_survived     = train_df["Fare"][train_df["Survived"] == 1]



# get average and std for fare of survived/not survived passengers

average_fare = DataFrame([fare_not_survived.mean(), fare_survived.mean()])

std_fare      = DataFrame([fare_not_survived.std(), fare_survived.std()])





average_fare.plot(yerr=std_fare,kind='bar',legend=False)


count_sibsp_nan_test = test_df["SibSp"].isnull().sum()

print(count_sibsp_nan_test)

count_sibsp_nan_train = train_df['SibSp'].isnull().sum()

print(count_sibsp_nan_train)



count_parch_nan_test = test_df["Parch"].isnull().sum()

print(count_parch_nan_test)

count_parch_nan_train = train_df['Parch'].isnull().sum()

print(count_parch_nan_train)
embark_dummies_titanic  = pd.get_dummies(train_df['Embarked'])

embark_dummies_titanic.drop(['S'], axis=1)



embark_dummies_test  = pd.get_dummies(test_df['Embarked'])

embark_dummies_test.drop(['S'], axis=1)



train_df = train_df.join(embark_dummies_titanic)

test_df    = test_df.join(embark_dummies_test)



train_df = train_df.drop(['Embarked'], axis=1)

test_df = test_df.drop(['Embarked'], axis=1)
print(test_df.head(2))

#print(train_df[~train_df.applymap(np.isreal).all(1)])

print( np.argmin(test_df.applymap(np.isreal).all(1)))

#print(X_train)



X_train = train_df.drop("Survived",axis=1)

print( np.argmin(X_train.applymap(np.isreal).all(1)))

Y_train = train_df["Survived"]

X_test  = test_df.drop("PassengerId",axis=1).copy()

print( np.argmin(X_test.applymap(np.isreal).all(1)))

print('-----dataset----')

print(X_train.head(2))
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)



knn.fit(X_train, Y_train)



Y_pred = knn.predict(X_test)



print(knn.score(X_train, Y_train))

submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('titanic.csv', index=False)
# Random Forests

from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100)



random_forest.fit(X_train, Y_train)



Y_pred = random_forest.predict(X_test)



random_forest.score(X_train, Y_train)
#SVM---->>> SVR

from sklearn.svm import SVC, LinearSVC, SVR

clf = SVR(C=1.0, epsilon=0.2)

clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_test)



clf.score(X_train, Y_train)
#SVM---->>> SVC

from sklearn.svm import SVC, LinearSVC, SVR

clf = SVC()

clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_test)



clf.score(X_train, Y_train)