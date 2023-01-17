# This Python 3 environment comes with many helpful analytics libraries installed

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # data visualization

sns.set_style('whitegrid')

import matplotlib.pyplot as plt # data visualization

%matplotlib inline
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
titanic_train = pd.read_csv("../input/train.csv")

titanic_test = pd.read_csv("../input/test.csv")
print("Train: rows:{} columns:{}".format(titanic_train.shape[0], titanic_train.shape[1]))
print("Test Data Shape",titanic_test.shape)
titanic_train.head()
titanic_train.tail()
titanic_test.head()
titanic_test.tail()
print("Total Number of passagner on Titanic (from training data):", str(len(titanic_train)))
sns.countplot(x="Survived", data=titanic_train)
sns.countplot(x="Survived", hue = 'Sex', data=titanic_train)
sns.countplot(x = "Survived", hue = "Pclass", data = titanic_train)
titanic_train['Age'].plot.hist()
titanic_train['Fare'].plot.hist(bins = 20, figsize = (10,5))
titanic_train.info()

print("----------------------------")

titanic_test.info()
sns.countplot(x= "SibSp", data = titanic_train)
sns.countplot(x = "Parch", data = titanic_train)
titanic_train.isnull().sum()
sns.heatmap(titanic_train.isnull(), yticklabels=False, cmap = 'viridis')
sns.boxplot(x = 'Pclass', y = 'Age', data = titanic_train)
print("Training dataset columns:",titanic_train.columns)

print("-------------------------------")

print("Training dataset columns:",titanic_test.columns)
# Drop unnecessary columns, these columns won't be useful in analysis and prediction

titanic_train = titanic_train.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)

titanic_test = titanic_test.drop(['Name','Ticket','Cabin'], axis=1)
print("Training dataset columns:",titanic_train.columns)

print("-------------------------------")

print("Training dataset columns:",titanic_test.columns)
# Only in titanic_train dataset, fill the two missing values with the most occurred value, which is "S".

titanic_train["Embarked"] = titanic_train["Embarked"].fillna("S")



# plot

sns.factorplot('Embarked','Survived', data=titanic_train, size=5, aspect=3)



fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5))



# sns.factorplot('Embarked', data=titanic_df, kind='count', order=['S','C','Q'], ax=axis1)

# sns.factorplot('Survived',hue="Embarked",data=titanic_df,kind='count',order=[1,0],ax=axis2)

sns.countplot(x='Embarked', data=titanic_train, ax=axis1)

sns.countplot(x='Survived', hue="Embarked", data=titanic_train, order=[1,0], ax=axis2)



# group by embarked, and get the mean for survived passengers for each value in Embarked

embark_perc = titanic_train[["Embarked", "Survived"]].groupby(['Embarked'],as_index=False).mean()

sns.barplot(x='Embarked', y='Survived', data=embark_perc,order=['S','C','Q'],ax=axis3)
'''

Either to consider Embarked column in predictions, and remove "S" dummy variable, 

and leave "C" & "Q", since they seem to have a good rate for Survival.



OR, don't create dummy variables for Embarked column, just drop it, 

because logically, Embarked doesn't seem to be useful in prediction.

'''



embark_dummies_titanic_train  = pd.get_dummies(titanic_train['Embarked'])

embark_dummies_titanic_train.drop(['S'], axis=1, inplace=True)



embark_dummies_titanic_test  = pd.get_dummies(titanic_test['Embarked'])

embark_dummies_titanic_test.drop(['S'], axis=1, inplace=True)
titanic_train = titanic_train.join(embark_dummies_titanic_train)

titanic_test  = titanic_test.join(embark_dummies_titanic_test)
titanic_train.drop(['Embarked'], axis=1,inplace=True)

titanic_test.drop(['Embarked'], axis=1,inplace=True)
titanic_train.head()
titanic_test.head()
# Only for titanic_test, since there is a missing "Fare" values

titanic_test["Fare"].fillna(titanic_test["Fare"].median(), inplace=True)
# Convert from float to int

titanic_train['Fare'] = titanic_train['Fare'].astype(int)

titanic_test['Fare'] = titanic_test['Fare'].astype(int)
# Get fare for survived & didn't survive passengers 

fare_not_survived = titanic_train["Fare"][titanic_train["Survived"] == 0]

fare_survived     = titanic_train["Fare"][titanic_train["Survived"] == 1]



# Get average and std for fare of survived/not survived passengers

avg_fare = pd.DataFrame([fare_not_survived.mean(), fare_survived.mean()])

std_fare = pd.DataFrame([fare_not_survived.std(), fare_survived.std()])
# plot

titanic_train['Fare'].plot(kind='hist', figsize=(15,3),bins=100, xlim=(0,50))



avg_fare.index.names = std_fare.index.names = ["Survived"]

avg_fare.plot(yerr=std_fare,kind='bar',legend=False)
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))

axis1.set_title('Original Age Values - Titanic')

axis2.set_title('New Age Values - Titanic')



# axis3.set_title('Original Age values - Test')

# axis4.set_title('New Age values - Test')



# Get Average, STD, and Number of NaN values in titanic_train

avg_age_titanic_train = titanic_train["Age"].mean()

std_age_titanic_train = titanic_train["Age"].std()

count_nan_age_titanic_train = titanic_train["Age"].isnull().sum()



# Get Average, STD, and Number of NaN values in titanic_test

avg_age_titanic_test = titanic_test["Age"].mean()

std_age_titanic_test = titanic_test["Age"].std()

count_nan_age_titanic_test = titanic_test["Age"].isnull().sum()



# Generate random numbers between (mean - std) & (mean + std)

rand_1 = np.random.randint(avg_age_titanic_train - std_age_titanic_train, avg_age_titanic_train + std_age_titanic_train, size = count_nan_age_titanic_train)

rand_2 = np.random.randint(avg_age_titanic_test - std_age_titanic_test, avg_age_titanic_test + std_age_titanic_test, size = count_nan_age_titanic_test)



# plot original Age values

# NOTE: drop all null values, and convert to int

titanic_train['Age'].dropna().astype(int).hist(bins=70, ax=axis1)

# test_df['Age'].dropna().astype(int).hist(bins=70, ax=axis1)



# fill NaN values in Age column with random values generated

titanic_train["Age"][np.isnan(titanic_train["Age"])] = rand_1

titanic_test["Age"][np.isnan(titanic_test["Age"])] = rand_2



# Convert from float to int

titanic_train['Age'] = titanic_train['Age'].astype(int)

titanic_test['Age'] = titanic_test['Age'].astype(int)

        

# plot new Age Values

titanic_train['Age'].hist(bins=70, ax=axis2)

#titanic_test['Age'].hist(bins=70, ax=axis4)
# peaks for survived/not survived passengers by their age

facet = sns.FacetGrid(titanic_train, hue="Survived",aspect=4)

facet.map(sns.kdeplot,'Age',shade= True)

facet.set(xlim=(0, titanic_train['Age'].max()))

facet.add_legend()



# average survived passengers by age

fig, axis1 = plt.subplots(1,1,figsize=(18,4))

avg_age = titanic_train[["Age", "Survived"]].groupby(['Age'],as_index=False).mean()

sns.barplot(x='Age', y='Survived', data=avg_age)
""" 

Instead of having two columns Parch & SibSp, We can have only one column represent 

if the passenger had any family member aboard or not,

Meaning, if having any family member(whether parent, brother, ...etc) will increase chances of Survival or not.

"""

# make changes with training dataset

titanic_train['Family'] =  titanic_train["Parch"] + titanic_train["SibSp"]

titanic_train['Family'].loc[titanic_train['Family'] > 0] = 1

titanic_train['Family'].loc[titanic_train['Family'] == 0] = 0



# make changes with test dataset

titanic_test['Family'] =  titanic_test["Parch"] + titanic_test["SibSp"]

titanic_test['Family'].loc[titanic_test['Family'] > 0] = 1

titanic_test['Family'].loc[titanic_test['Family'] == 0] = 0
# Now we will drop Parch & SibSp

titanic_train = titanic_train.drop(['SibSp','Parch'], axis=1)

titanic_test = titanic_test.drop(['SibSp','Parch'], axis=1)
# Plot

fig, (axis1,axis2) = plt.subplots(1,2,sharex=True,figsize=(10,5))



# sns.factorplot('Family',data=titanic_train,kind='count',ax=axis1)

sns.countplot(x='Family', data=titanic_train, order=[1,0], ax=axis1)



# average of survived for those who had/didn't have any family member

family_perc = titanic_train[["Family", "Survived"]].groupby(['Family'],as_index=False).mean()

sns.barplot(x='Family', y='Survived', data=family_perc, order=[1,0], ax=axis2)



axis1.set_xticklabels(["With Family","Alone"], rotation=0)
# As we see, children(age < 16) on aboard seem to have a high chances for Survival.

# So, we can classify passengers as males, females, and child



def get_person(passenger):

    age,sex = passenger

    return 'child' if age < 16 else sex

    

titanic_train['Person'] = titanic_train[['Age','Sex']].apply(get_person,axis=1)

titanic_test['Person'] = titanic_test[['Age','Sex']].apply(get_person,axis=1)



# No need to use Sex column since we created Person column

titanic_train.drop(['Sex'],axis=1,inplace=True)

titanic_test.drop(['Sex'],axis=1,inplace=True)
# Create dummy variables for Person column, & drop Male as it has the lowest average of survived passengers

person_dummies_titanic_train  = pd.get_dummies(titanic_train['Person'])

person_dummies_titanic_train.columns = ['Child','Female','Male']

person_dummies_titanic_train.drop(['Male'], axis=1, inplace=True)



person_dummies_titanic_test  = pd.get_dummies(titanic_test['Person'])

person_dummies_titanic_test.columns = ['Child','Female','Male']

person_dummies_titanic_test.drop(['Male'], axis=1, inplace=True)



titanic_train = titanic_train.join(person_dummies_titanic_train)

titanic_test = titanic_test.join(person_dummies_titanic_test)



fig, (axis1,axis2) = plt.subplots(1,2,figsize=(10,5))



# sns.factorplot('Person',data=titanic_train,kind='count',ax=axis1)

sns.countplot(x='Person', data=titanic_train, ax=axis1)



# average of survived for each Person(male, female, or child)

person_perc = titanic_train[["Person", "Survived"]].groupby(['Person'],as_index=False).mean()

sns.barplot(x='Person', y='Survived', data=person_perc, ax=axis2, order=['male','female','child'])
titanic_train.drop(['Person'],axis=1,inplace=True)

titanic_test.drop(['Person'],axis=1,inplace=True)
# sns.factorplot('Pclass',data=titanic_train,kind='count',order=[1,2,3])

sns.factorplot('Pclass','Survived',order=[1,2,3], data=titanic_train, size=5)



# create dummy variables for Pclass column, & drop 3rd class as it has the lowest average of survived passengers

pclass_dummies_titanic_train  = pd.get_dummies(titanic_train['Pclass'])

pclass_dummies_titanic_train.columns = ['Class_1','Class_2','Class_3']

pclass_dummies_titanic_train.drop(['Class_3'], axis=1, inplace=True)



pclass_dummies_titanic_test  = pd.get_dummies(titanic_test['Pclass'])

pclass_dummies_titanic_test.columns = ['Class_1','Class_2','Class_3']

pclass_dummies_titanic_test.drop(['Class_3'], axis=1, inplace=True)



titanic_train = titanic_train.join(pclass_dummies_titanic_train)

titanic_test = titanic_test.join(pclass_dummies_titanic_test)
titanic_train.drop(['Pclass'],axis=1,inplace=True)

titanic_test.drop(['Pclass'],axis=1,inplace=True)
titanic_train.head()
titanic_test.head()
# Descriptive statistics for each column

titanic_train.describe()
titanic_test.describe()
X_train = titanic_train.drop("Survived",axis=1)

Y_train = titanic_train["Survived"]

X_test  = titanic_test.drop("PassengerId",axis=1).copy()
logreg = LogisticRegression()



logreg.fit(X_train, Y_train)



Y_pred = logreg.predict(X_test)



acc_log = logreg.score(X_train, Y_train)



acc_log
# Support Vector Machines



svc = SVC()



svc.fit(X_train, Y_train)



Y_pred = svc.predict(X_test)



acc_svc = svc.score(X_train, Y_train)



acc_svc
# Random Forests



random_forest = RandomForestClassifier(n_estimators=100)



random_forest.fit(X_train, Y_train)



Y_pred = random_forest.predict(X_test)



acc_random_forest = random_forest.score(X_train, Y_train)



acc_random_forest
# Get Correlation Coefficient for each feature using Logistic Regression

coeff_df = pd.DataFrame(titanic_train.columns.delete(0))

coeff_df.columns = ['Features']

coeff_df["Coefficient Estimate"] = pd.Series(logreg.coef_[0])



# preview

coeff_df
models = pd.DataFrame({

    'Model': ['Logistic Regression', 'Support Vector Machines',

              'Random Forest'],

    'Score': [acc_log, acc_svc,

              acc_random_forest]})

models.sort_values(by='Score', ascending=False)
submission = pd.DataFrame({

        "PassengerId": titanic_test["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('titanic_submission1.csv', index=False)