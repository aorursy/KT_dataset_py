

#import pandas and numpy

import numpy as np 

import pandas as pd 

from pandas import Series,DataFrame

# for data visualization

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



#Machine Learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB





# create a dataframe 

titanic_train_df = pd.read_csv('../input/train.csv')

titanic_test_df = pd.read_csv('../input/test.csv')

combine = [titanic_train_df, titanic_test_df]

# to get the first 5 values from the dataframe of titanic_train_df

titanic_train_df.head(5)

#to get the information about the dataset ,we can use the below function

titanic_train_df.info()

print("*"*30)

titanic_test_df.info()
# one of the useful function in Pandas to generatwe descriptive statistics

#distribution of numerical feature values across the samples

titanic_train_df.describe()

#it reflect the distribution of categorical features in the dataset

titanic_train_df.describe(include=['O'])
#groupby  help us in combining statistics about the dataframe,(sort_values)Sort by the values along either axis



titanic_train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
titanic_train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
titanic_train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
titanic_train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
gan = sns.FacetGrid(titanic_train_df, col='Survived')

gan.map(plt.hist, 'Age', bins=20)
grid = sns.FacetGrid(titanic_train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();
grid = sns.FacetGrid(titanic_train_df, row='Embarked', size=2.2, aspect=1.6)

grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')

grid.add_legend()
grid = sns.FacetGrid(titanic_train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)

grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)

grid.add_legend()
print("Before", titanic_train_df.shape, titanic_test_df.shape, combine[0].shape, combine[1].shape)



titanic_train_df = titanic_train_df.drop(['Ticket', 'Cabin'], axis=1)

titanic_test_df = titanic_test_df.drop(['Ticket', 'Cabin'], axis=1)

combine = [titanic_train_df , titanic_test_df]



"After", titanic_train_df.shape, titanic_test_df.shape, combine[0].shape, combine[1].shape
for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



pd.crosstab(titanic_train_df['Title'], titanic_train_df['Sex'])
for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    

titanic_train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
#convert the categorical titles to ordinal.

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in combine:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)



titanic_train_df.head()
titanic_train_df = titanic_train_df.drop(['Name', 'PassengerId'], axis=1)

titanic_test_df = titanic_test_df.drop(['Name'], axis=1)

combine = [titanic_train_df, titanic_test_df]

titanic_train_df.shape, titanic_test_df.shape
for dataset in combine:

    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)



titanic_train_df.head()
grid = sns.FacetGrid(titanic_train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend()
#Preperation step:

#Let start by preparing an empty array to contain guessed Age values based on Pclass x Gender combinations.

guess_ages = np.zeros((2,3))

guess_ages
#Now lets iterate over Sex (0 or 1) and Pclass (1, 2, 3) to calculate guessed values of Age for the six combinations.

for dataset in combine:

    for i in range(0, 2):

        for j in range(0, 3):

            guess_df = dataset[(dataset['Sex'] == i) & \

                                  (dataset['Pclass'] == j+1)]['Age'].dropna()



            age_guess = guess_df.median()



            # Convert random age float to nearest .5 age

            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

            

    for i in range(0, 2):

        for j in range(0, 3):

            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\

                    'Age'] = guess_ages[i,j]



    dataset['Age'] = dataset['Age'].astype(int)



titanic_train_df.head()

















#Let us create Age bands and determine correlations with Survived.



titanic_train_df['AgeBand'] = pd.cut(titanic_train_df['Age'], 5)

titanic_train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)







#Let us replace Age with ordinals based on these bands.



for dataset in combine:    

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age']

titanic_train_df.head()







#We can now remove the AgeBand feature.



titanic_train_df = titanic_train_df.drop(['AgeBand'], axis=1)

combine = [titanic_train_df, titanic_test_df]

titanic_train_df.head()





 #Create new feature combining existing features

# create a new feature for FamilySize which combines Parch and SibSp. This will encourage to drop Parch and SibSp from our datasets.



for dataset in combine:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1



titanic_train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)







                                                                                                                                                             




#We can create another feature called IsAlone.



for dataset in combine:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1



titanic_train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
#Let us drop Parch, SibSp, and FamilySize features in favor of IsAlone.



titanic_train_df = titanic_train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

titanic_test_df= titanic_test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

combine = [titanic_train_df, titanic_test_df]



titanic_train_df.head()
# create an artificial feature combining Pclass and Age.

for dataset in combine:

    dataset['Age*Class'] = dataset.Age * dataset.Pclass



titanic_train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)









freq_port = titanic_train_df.Embarked.dropna().mode()[0]

freq_port
for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

    

titanic_train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)



titanic_train_df.head()
titanic_test_df['Fare'].fillna(titanic_test_df['Fare'].dropna().median(), inplace=True)

titanic_test_df.head()
titanic_train_df['FareBand'] = pd.qcut(titanic_train_df['Fare'], 4)

titanic_train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
for dataset in combine:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)



titanic_train_df = titanic_train_df.drop(['FareBand'], axis=1)

combine = [titanic_train_df, titanic_test_df]

    

titanic_train_df.head(10)
X_train = titanic_train_df.drop("Survived", axis=1)

Y_train = titanic_train_df["Survived"]

X_test  = titanic_test_df.drop("PassengerId", axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape
logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

acc_log
# Random Forest



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

acc_random_forest
submission = pd.DataFrame({

        "PassengerId": titanic_test_df["PassengerId"],

        "Survived": Y_pred

    })

# submission.to_csv('../output/submission.csv', index=False)
#in this way prediction of the titanic survivor is done