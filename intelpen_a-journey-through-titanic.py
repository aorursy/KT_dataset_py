# Imports



# pandas

import pandas as pd

from pandas import Series,DataFrame



# numpy, matplotlib, seaborn

import numpy as np

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
# get titanic & test csv files as a DataFrame

titanic_df = pd.read_csv("../input/train.csv")

test_df    = pd.read_csv("../input/test.csv")



# preview the data

titanic_df.head()
titanic_df.info()

print("----------------------------")

test_df.info()
# drop less likely to count columns and overwrite over the initial dataframe

titanic_df = titanic_df.drop(['PassengerId','Name','Ticket'], axis=1)

test_df    = test_df.drop(['Name','Ticket'], axis=1)
# Embarked



# only in titanic_df, fill the two missing values with the most occurred value, which is "S".

titanic_df["Embarked"] = titanic_df["Embarked"].fillna("S")

titanic_df.loc[titanic_df["Embarked"] == "S", "Embarked"] = 0



titanic_df.loc[titanic_df["Embarked"] == "C", "Embarked"] = 1



titanic_df.loc[titanic_df["Embarked"] == "Q", "Embarked"] = 2



test_df["Embarked"] = test_df["Embarked"].fillna("S")

test_df.loc[test_df["Embarked"] == "S", "Embarked"] = 0



test_df.loc[test_df["Embarked"] == "C", "Embarked"] = 1



test_df.loc[test_df["Embarked"] == "Q", "Embarked"] = 2





# plot

sns.factorplot('Embarked','Survived', data=titanic_df,size=4,aspect=3)



fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5))



# sns.factorplot('Embarked',data=titanic_df,kind='count',order=['S','C','Q'],ax=axis1)

# sns.factorplot('Survived',hue="Embarked",data=titanic_df,kind='count',order=[1,0],ax=axis2)

sns.countplot(x='Embarked', data=titanic_df, ax=axis1)

sns.countplot(x='Survived', hue="Embarked", data=titanic_df, order=[1,0], ax=axis2)



# group by embarked, and get the mean for survived passengers for each value in Embarked

embark_perc = titanic_df[["Embarked", "Survived"]].groupby(['Embarked'],as_index=False).mean()

sns.barplot(x='Embarked', y='Survived', data=embark_perc,order=['S','C','Q'],ax=axis3)



# Either to consider Embarked column in predictions,

# and remove "S" dummy variable, 

# and leave "C" & "Q", since they seem to have a good rate for Survival.

# Fare



# only for test_df, since there is a missing "Fare" values

test_df["Fare"].fillna(test_df["Fare"].median(), inplace=True)



# convert from float to int

titanic_df['Fare'] = titanic_df['Fare'].astype(int)

test_df['Fare']    = test_df['Fare'].astype(int)



# get fare for survived & didn't survive passengers 

fare_not_survived = titanic_df["Fare"][titanic_df["Survived"] == 0]

fare_survived     = titanic_df["Fare"][titanic_df["Survived"] == 1]



# get average and std for fare of survived/not survived passengers

avgerage_fare = DataFrame([fare_not_survived.mean(), fare_survived.mean()])

std_fare      = DataFrame([fare_not_survived.std(), fare_survived.std()])



# plot

titanic_df['Fare'].plot(kind='hist', figsize=(15,3),bins=100, xlim=(0,50))



avgerage_fare.index.names = std_fare.index.names = ["Survived"]

avgerage_fare.plot(yerr=std_fare,kind='bar',legend=False)
# Age 



fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))

axis1.set_title('Original Age values - Titanic')

axis2.set_title('New Age values - Titanic')



# axis3.set_title('Original Age values - Test')

# axis4.set_title('New Age values - Test')



# get average, std, and number of NaN values in titanic_df

average_age_titanic   = titanic_df["Age"].mean()

std_age_titanic       = titanic_df["Age"].std()

count_nan_age_titanic = titanic_df["Age"].isnull().sum()



# get average, std, and number of NaN values in test_df

average_age_test   = test_df["Age"].mean()

std_age_test       = test_df["Age"].std()

count_nan_age_test = test_df["Age"].isnull().sum()



# generate random numbers between (mean - std) & (mean + std)

rand_1 = np.random.randint(average_age_titanic - std_age_titanic, average_age_titanic + std_age_titanic, size = count_nan_age_titanic)

rand_2 = np.random.randint(average_age_test - std_age_test, average_age_test + std_age_test, size = count_nan_age_test)



# plot original Age values

# NOTE: drop all null values, and convert to int

titanic_df['Age'].dropna().astype(int).hist(bins=70, ax=axis1)

# test_df['Age'].dropna().astype(int).hist(bins=70, ax=axis1)



# fill NaN values in Age column with random values generated

titanic_df["Age"][np.isnan(titanic_df["Age"])] = average_age_titanic

test_df["Age"][np.isnan(test_df["Age"])] = average_age_test



# convert from float to int

titanic_df['Age'] = titanic_df['Age'].astype(int)

test_df['Age']    = test_df['Age'].astype(int)

        

# plot new Age Values

titanic_df['Age'].hist(bins=70, ax=axis2)

# test_df['Age'].hist(bins=70, ax=axis4)
# .... continue with plot Age column



# peaks for survived/not survived passengers by their age

facet = sns.FacetGrid(titanic_df, hue="Survived",aspect=4)

facet.map(sns.kdeplot,'Age',shade= True)

facet.set(xlim=(0, titanic_df['Age'].max()))

facet.add_legend()



# average survived passengers by age

fig, axis1 = plt.subplots(1,1,figsize=(18,4))

average_age = titanic_df[["Age", "Survived"]].groupby(['Age'],as_index=False).mean()

sns.barplot(x='Age', y='Survived', data=average_age)
# Cabin

# It has a lot of NaN values, so it won't cause a remarkable impact on prediction

titanic_df.drop("Cabin",axis=1,inplace=True)

test_df.drop("Cabin",axis=1,inplace=True)
# Pclass



# sns.factorplot('Pclass',data=titanic_df,kind='count',order=[1,2,3])

sns.factorplot('Pclass','Survived',order=[1,2,3], data=titanic_df,size=5)

titanic_df.loc[titanic_df["Sex"] == "male", "Sex"] = 0

titanic_df.loc[titanic_df["Sex"] == "female", "Sex"] = 1

test_df.loc[test_df["Sex"] == "male", "Sex"] = 0

test_df.loc[test_df["Sex"] == "female", "Sex"] = 1

titanic_df.head()
# define training and testing sets



X_train = titanic_df.drop("Survived",axis=1)

Y_train = titanic_df["Survived"]

X_test  = test_df.drop("PassengerId",axis=1).copy()

X_train.head()
# Logistic Regression



logreg = LogisticRegression()



logreg.fit(X_train, Y_train)



Y_pred = logreg.predict(X_test)



logreg.score(X_train, Y_train)
# Support Vector Machines



# svc = SVC()



# svc.fit(X_train, Y_train)



# Y_pred = svc.predict(X_test)



# svc.score(X_train, Y_train)
# Random Forests



random_forest = RandomForestClassifier(n_estimators=100)



random_forest.fit(X_train, Y_train)



Y_pred = random_forest.predict(X_test)



random_forest.score(X_train, Y_train)
# knn = KNeighborsClassifier(n_neighbors = 3)



# knn.fit(X_train, Y_train)



# Y_pred = knn.predict(X_test)



# knn.score(X_train, Y_train)
# Gaussian Naive Bayes



# gaussian = GaussianNB()



# gaussian.fit(X_train, Y_train)



# Y_pred = gaussian.predict(X_test)



# gaussian.score(X_train, Y_train)
# get Correlation Coefficient for each feature using Logistic Regression

coeff_df = DataFrame(titanic_df.columns.delete(0))

coeff_df.columns = ['Features']

coeff_df["Coefficient Estimate"] = pd.Series(logreg.coef_[0])



# preview

coeff_df
submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('titanic.csv', index=False)