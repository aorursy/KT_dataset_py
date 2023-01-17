# Pandas

import pandas as pd 

from pandas import Series,DataFrame

# Numpy

import numpy as np # linear algebra

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline







# Machione Learning 

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC,LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB
# Get Titanic data and separate it in test and train data 

train_titanic  = pd.read_csv("../input/train.csv")

test_titanic = pd.read_csv("../input/test.csv")
# Data Preview

train_titanic.head()

print('///////////////////')

train_titanic.info()

print('///////////////////')

test_titanic.info()
test_titanic.head()
# Drop unneccessary columns as this is not usefull in analysis and predictions

train_titanic = train_titanic.drop(['PassengerId','Name','Ticket'],axis= 1)

train_titanic.head()

test_titanic = test_titanic.drop(['Name','Ticket'],axis = 1)

test_titanic.head()
# Embarked

train_titanic.count()

train_titanic.describe(include=['O'])

df = pd.DataFrame(train_titanic)

df.describe()
# Missing Value filled with most occurence value in Embarked column

train_titanic["Embarked"] = train_titanic["Embarked"].fillna("S")

train_titanic.count()
# Plotting of data

sns.catplot('Embarked','Survived',data= train_titanic,height = 4,aspect = 1)

fig,(axis1,axis2,axis3)= plt.subplots(1,3,figsize=(15,5))
#sns.factorplot('Embarked',data= train_titanic,kind='count',order=['S','C','Q'],ax=axis1)

#sns.factorplot('Survived',hue="Embarked",data= train_titanic,kind = 'count',order= [1,0],ax= axis2)

sns.countplot(x='Embarked',data = train_titanic,ax= axis1)

sns.countplot(x='Survived',hue= "Embarked",data = train_titanic,order=[1,0],ax= axis2)
# group by embarked, and get the mean for survived passengers for each value in Embarked

embark_perc =train_titanic[["Embarked","Survived"]].groupby(['Embarked'],as_index=False).mean()

sns.barplot(x='Embarked', y='Survived', data=embark_perc,order=['S','C','Q'],ax=axis3)
# Either to consider Embarked column in predictions,

# and remove "S" dummy variable, 

# and leave "C" & "Q", since they seem to have a good rate for Survival.



# OR, don't create dummy variables for Embarked column, just drop it, 

# because logically, Embarked doesn't seem to be useful in prediction.
embark_dummies_titanic = pd.get_dummies(train_titanic['Embarked'])

embark_dummies_titanic.drop(['S'],axis = 1,inplace= True)



embark_dummies_test  = pd.get_dummies(test_titanic['Embarked'])

embark_dummies_test.drop(['S'], axis=1, inplace=True)

print(embark_dummies_titanic)

print(train_titanic)

train_titanic = train_titanic.join(embark_dummies_titanic)

print(train_titanic)

test_titanic    = test_titanic.join(embark_dummies_test)

print(test_titanic)

train_titanic.drop(['Embarked'], axis=1,inplace=True)

test_titanic.drop(['Embarked'], axis=1,inplace=True)
# Fare

# Only for test_titanic,since there is a missing "Fare" values

test_titanic["Fare"].fillna(test_titanic["Fare"].median(),inplace= True)
# Convert from Float to int 

train_titanic['Fare'] = train_titanic['Fare'].astype(int)

test_titanic['Fare'] = test_titanic['Fare'].astype(int)
# get fare for survived and unsurvived Passenger 

fare_not_survived = train_titanic["Fare"][train_titanic["Survived"]==0]

fare_survived = train_titanic["Fare"][train_titanic["Survived"]==1]
# get std and avg for survived and unsurvived passenger

avg_fare = DataFrame([fare_not_survived.mean(),fare_survived.mean()])

std_fare = DataFrame([fare_not_survived.std(),fare_survived.std()])
# plot

train_titanic['Fare'].plot(kind= 'hist',figsize= (15,3),bins = 100,xlim= (0,50))

avg_fare.index.names = std_fare.index.names = ["Survived"]

avg_fare.plot(yerr= std_fare,kind= 'bar',legend= False)
# age

fig,(axis1,axis2)=plt.subplots(1,2,figsize= (15,4))

axis1.set_title('Original Value titanic')

axis2.set_title('New Age values - Titanic')
# get average, std, and number of NaN values in train_titanic

avg_age_titanic_train = train_titanic["Age"].mean()

print(avg_age_titanic_train)

std_age_titanic_train = train_titanic["Age"].mean()

print(std_age_titanic_train)

count_nan_age_titanic_train = train_titanic["Age"].isnull().sum()

print(count_nan_age_titanic_train)

# get average, std, and number of NaN values in test_titanic

avg_age_titanic_test = test_titanic["Age"].mean()

print(avg_age_titanic_test)

std_age_titanic_test = test_titanic["Age"].mean()

print(std_age_titanic_test)

count_nan_age_titanic_test = test_titanic["Age"].isnull().sum()

print(count_nan_age_titanic_test)

# generate random numbers between (mean - std) & (mean + std)

rand1 = np.random.randint(avg_age_titanic_train - std_age_titanic_train,avg_age_titanic_train +std_age_titanic_train,size = count_nan_age_titanic_train)

print (rand1)

rand2 = np.random.randint(avg_age_titanic_test - std_age_titanic_test,avg_age_titanic_test +std_age_titanic_train,size = count_nan_age_titanic_test)

print (rand2)
# plot original Age values

# NOTE: drop all null values, and convert to int

#train_titanic['Age'].dropna().astype(int).hist(bins=70, ax=axis1)

#test_titanic['Age'].dropna().astype(int).hist(bins=70,ax= axis1)
train_titanic.info()

print(rand1)
# fill NaN values in Age column with random values generated

train_titanic["Age"][np.isnan(train_titanic["Age"])] = rand1

test_titanic["Age"][np.isnan(test_titanic["Age"])] = rand2
# convert from float to int

train_titanic['Age'] = train_titanic['Age'].astype(int)

test_titanic['Age']    = test_titanic['Age'].astype(int)
# plot new Age Values

#train_titanic['Age'].hist(bins=70, ax=axis2)

# test_df['Age'].hist(bins=70, ax=axis4)

train_titanic.info()
# Peaks for survived and unsurvived and unsurvived passenger by their age

facet = sns.FacetGrid(train_titanic,hue = "Survived",aspect = 4)

facet.map(sns.kdeplot,'Age',shade = True)

facet.set(xlim= (0,train_titanic['Age'].max()))

facet.add_legend()
# average survived passengers by age

fig, axis1 = plt.subplots(1,1,figsize=(18,4))

average_age = train_titanic[["Age", "Survived"]].groupby(['Age'],as_index=False).mean()

sns.barplot(x='Age', y='Survived', data=average_age)
# Cabin

# It has a lot of NaN values, so it won't cause a remarkable impact on prediction

train_titanic.drop("Cabin",axis=1,inplace=True)

test_titanic.drop("Cabin",axis=1,inplace=True)
# Family



# Instead of having two columns Parch & SibSp, 

# we can have only one column represent if the passenger had any family member aboard or not,

# Meaning, if having any family member(whether parent, brother, ...etc) will increase chances of Survival or not.
train_titanic['Family'] =  train_titanic["Parch"] + train_titanic["SibSp"]

train_titanic['Family'].loc[train_titanic['Family'] > 0] = 1

train_titanic['Family'].loc[train_titanic['Family'] == 0] = 0
test_titanic['Family'] =  test_titanic["Parch"] + test_titanic["SibSp"]

test_titanic['Family'].loc[test_titanic['Family'] > 0] = 1

test_titanic['Family'].loc[test_titanic['Family'] == 0] = 0
# drop Parch & SibSp

train_titanic = train_titanic.drop(['SibSp','Parch'], axis=1)

test_titanic    = test_titanic.drop(['SibSp','Parch'], axis=1)
# plot

fig, (axis1,axis2) = plt.subplots(1,2,sharex=True,figsize=(10,5))
# sns.factorplot('Family',data=titanic_df,kind='count',ax=axis1)

sns.countplot(x='Family', data=train_titanic, order=[1,0], ax=axis1)
# average of survived for those who had/didn't have any family member

family_perc = train_titanic[["Family", "Survived"]].groupby(['Family'],as_index=False).mean()

sns.barplot(x='Family', y='Survived', data=family_perc, order=[1,0], ax=axis2)

axis1.set_xticklabels(["With Family","Alone"], rotation=0)
# Sex



# As we see, children(age < ~16) on aboard seem to have a high chances for Survival.

# So, we can classify passengers as males, females, and child
def get_person(passenger):

    age,sex = passenger

    return 'child' if age < 16 else sex
train_titanic['Person'] = train_titanic[['Age','Sex']].apply(get_person,axis=1)

test_titanic['Person']    = test_titanic[['Age','Sex']].apply(get_person,axis=1)
# No need to use Sex column since we created Person column

train_titanic.drop(['Sex'],axis=1,inplace=True)

test_titanic.drop(['Sex'],axis=1,inplace=True)
train_titanic.info()
# create dummy variables for Person column, & drop Male as it has the lowest average of survived passengers

person_dummies_titanic = pd.get_dummies(train_titanic['Person'])

person_dummies_titanic.columns = ['Child','Female','Male']

person_dummies_titanic.drop(['Male'], axis=1, inplace=True)
person_dummies_test  = pd.get_dummies(test_titanic['Person'])

person_dummies_test.columns = ['Child','Female','Male']

person_dummies_test.drop(['Male'], axis=1, inplace=True)
train_titanic = train_titanic.join(person_dummies_titanic)

test_titanic = test_titanic.join(person_dummies_test)
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(10,5))
# sns.factorplot('Person',data=train_titanic,kind='count',ax=axis1)

sns.countplot(x='Person', data=train_titanic, ax=axis1)
# average of survived for each Person(male, female, or child)

person_perc = train_titanic[["Person","Survived"]].groupby(['Person'],as_index= False).mean()

sns.barplot(x='Person', y='Survived', data=person_perc, ax=axis2, order=['male','female','child'])
train_titanic.drop(['Person'],axis=1,inplace=True)

test_titanic.drop(['Person'],axis=1,inplace=True)
# Pclass



# sns.factorplot('Pclass',data=titanic_df,kind='count',order=[1,2,3])

sns.factorplot('Pclass','Survived',order=[1,2,3], data=train_titanic,size=5)



# create dummy variables for Pclass column, & drop 3rd class as it has the lowest average of survived passengers

pclass_dummies_titanic  = pd.get_dummies(train_titanic['Pclass'])

print(pclass_dummies_titanic)

pclass_dummies_titanic.columns = ['Class_1','Class_2','Class_3']

print(pclass_dummies_titanic)

pclass_dummies_titanic.drop(['Class_3'], axis=1, inplace=True)

print(pclass_dummies_titanic)



pclass_dummies_test  = pd.get_dummies(test_titanic['Pclass'])

pclass_dummies_test.columns = ['Class_1','Class_2','Class_3']

pclass_dummies_test.drop(['Class_3'], axis=1, inplace=True)



train_titanic.drop(['Pclass'],axis=1,inplace=True)

test_titanic.drop(['Pclass'],axis=1,inplace=True)



train_titanic = train_titanic.join(pclass_dummies_titanic)

test_titanic    = test_titanic.join(pclass_dummies_test)
# define training and testing sets



X_train = train_titanic.drop("Survived",axis=1)



Y_train = train_titanic["Survived"]

X_test = test_titanic.drop("PassengerId",axis=1).copy()



X_train.info()

X_test.info()
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

#knn



# knn = KNeighborsClassifier(n_neighbors = 3)



#  knn.fit(X_train, Y_train)



#  Y_pred = knn.predict(X_test)



#  knn.score(X_train, Y_train)
 # Gaussian

    

#  gaussian = GaussianNB()



#  gaussian.fit(X_train, Y_train)



#  Y_pred = gaussian.predict(X_test)



#  gaussian.score(X_train, Y_train)
# get Correlation Coefficient for each feature using Logistic Regression

coeff_df = DataFrame(train_titanic.columns.delete(0))

coeff_df.columns = ['Features']

coeff_df["Coefficient Estimate"] = pd.Series(logreg.coef_[0])
# Preview

coeff_df
submission = pd.DataFrame({

        "PassengerId": test_titanic["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('Save_titanic.csv', index=False)