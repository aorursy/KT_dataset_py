# Imports



# pandas

import pandas as pd

from pandas import Series,DataFrame



# numpy, matplotlib, seaborn

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

#matplotlib inline



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
# drop unnecessary columns, these columns won't be useful in analysis and prediction

titanic_df = titanic_df.drop(['PassengerId','Name','Ticket'], axis=1)

test_df    = test_df.drop(['Name','Ticket'], axis=1)
# Embarked



# only in titanic_df, fill the two missing values with the most occurred value, which is "S".

titanic_df["Embarked"] = titanic_df["Embarked"].fillna("S")

test_df["Embarked"] = test_df["Embarked"].fillna("S")

#create a new column for Southampton

def get_southampton(passenger):

    Embarked = passenger

    return 1 if (Embarked=='S').bool() else 0

titanic_df['Southampton']=titanic_df[['Embarked']].apply(get_southampton, axis=1)

test_df['Southampton']=test_df[['Embarked']].apply(get_southampton, axis=1)

#create a new column for Cherbourg

def get_cherbourg(passenger):

    Embarked = passenger

    return 1 if (Embarked=='C').bool() else 0

titanic_df['Cherbourg']=titanic_df[['Embarked']].apply(get_cherbourg, axis=1)

test_df['Cherbourg']=test_df[['Embarked']].apply(get_cherbourg, axis=1)

#create a new column for Queenstown

def get_queenstown(passenger):

    Embarked = passenger

    return 1 if (Embarked=='Q').bool() else 0

titanic_df['Queenstown']=titanic_df[['Embarked']].apply(get_queenstown, axis=1)

test_df['Queenstown']=test_df[['Embarked']].apply(get_queenstown, axis=1)
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
# Family





titanic_df = titanic_df.drop(['SibSp','Parch'], axis=1)

test_df    = test_df.drop(['SibSp','Parch'], axis=1)





# Pclass



#create a new column for first class

def get_firstclass(passenger):

    Pclass = passenger

    return 1 if (Pclass==1).bool() else 0

titanic_df['FirstClass']=titanic_df[['Pclass']].apply(get_firstclass, axis=1)

test_df['FirstClass']=test_df[['Pclass']].apply(get_firstclass, axis=1)



#create a new column for third class

def get_thirdclass(passenger):

    Pclass = passenger

    return 1 if (Pclass==3).bool() else 0

titanic_df['ThirdClass']=titanic_df[['Pclass']].apply(get_thirdclass, axis=1)

test_df['ThirdClass']=test_df[['Pclass']].apply(get_thirdclass, axis=1)

#create a new column for females

def get_female(passenger):

    Sex = passenger

    return 1 if (Sex=='female').bool() else 0

titanic_df['Female']=titanic_df[['Sex']].apply(get_female, axis=1)

test_df['Female']=test_df[['Sex']].apply(get_female, axis=1)

#create a new column for males

def get_male(passenger):

    Sex = passenger

    return 1 if (Sex=='male').bool() else 0

titanic_df['Male']=titanic_df[['Sex']].apply(get_male, axis=1)

test_df['Male']=test_df[['Sex']].apply(get_male, axis=1)
# define training and testing sets

titanic_df=titanic_df.drop(['Pclass'],axis=1)

test_df=test_df.drop(['Pclass'],axis=1)

titanic_df=titanic_df.drop(['Sex'],axis=1)

test_df=test_df.drop(['Sex'],axis=1)

titanic_df=titanic_df.drop(['Fare'],axis=1)

test_df=test_df.drop(['Fare'],axis=1)

titanic_df=titanic_df.drop(['Age'],axis=1)

test_df=test_df.drop(['Age'],axis=1)

titanic_df=titanic_df.drop(['Embarked'],axis=1)

test_df=test_df.drop(['Embarked'],axis=1)

X_train = titanic_df.drop("Survived",axis=1)

Y_train = titanic_df["Survived"]

X_test  = test_df.drop("PassengerId",axis=1).copy()
# Logistic Regression



logreg = LogisticRegression()



logreg.fit(X_train, Y_train)



Y_pred = logreg.predict(X_test)



logreg.score(X_train, Y_train)
# Support Vector Machines



svc = SVC()



svc.fit(X_train, Y_train)



Y_pred = svc.predict(X_test)



svc.score(X_train, Y_train)
# Random Forests



#random_forest = RandomForestClassifier(n_estimators=100)



#random_forest.fit(X_train, Y_train)



#Y_pred = random_forest.predict(X_test)



#random_forest.score(X_train, Y_train)
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