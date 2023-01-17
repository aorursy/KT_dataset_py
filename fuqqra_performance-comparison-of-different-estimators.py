import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib import style

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn import cross_validation



style.use('fivethirtyeight')

%matplotlib inline
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
#Drop the un-necasary columns:

train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True, axis=1)

test.drop(['Name', 'Ticket', 'Cabin'], inplace=True, axis=1)
# Check the count of values to see the missing values from both datasets:

print (train.info())

print ('=================================----------------')

print (test.info())
#Checking the embarkment values count to fill the nan.

train.Embarked.value_counts(), test.Embarked.value_counts()
#fill the missing values with the most redudant one.

train.Embarked.fillna('S', inplace=True)

test.Embarked.fillna('S', inplace=True)
#Relation between embarkment and Survival:

sns.countplot('Embarked', hue='Survived', data=train)

sns.factorplot('Embarked', 'Survived', data=train, size=3, aspect=2)
#create dummy variable for Embarked feature for our model:

train = pd.concat([train, pd.get_dummies(train.Embarked, prefix='embark')], axis=1)

train.drop('Embarked', inplace=True, axis=1)

test = pd.concat([test, pd.get_dummies(test.Embarked, prefix='embark')], axis=1)

test.drop('Embarked', inplace=True, axis=1)

train.head()
#Fare

train.Fare.mean(), train.Fare.median()
#lets fill it with mean value

train.Fare.fillna(train.Fare.mean(), inplace=True)

test.Fare.fillna(test.Fare.mean(), inplace=True)



train.Fare.hist()

print ('Spread of Fare:')
train.Age.mean(), train.Age.median()
train_age_mean = train.Age.mean()

train_age_std = train.Age.std()

train_age_count = train.Age.isnull().sum()



test_age_mean = test.Age.mean()

test_age_std = test.Age.std()

test_age_count = test.Age.isnull().sum()



rand_1 = np.random.randint(train_age_mean - train_age_std, train_age_mean + train_age_std, size=train_age_count)

rand_2 = np.random.randint(test_age_mean - test_age_std, test_age_mean + test_age_std, size=test_age_count)



train.Age[np.isnan(train.Age)] = rand_1

test.Age[np.isnan(test.Age)] = rand_2



train['Age'] = train.Age.astype(int)

test['Age'] = test.Age.astype(int)



#fill in the missing age values using the median value.

train.Age.hist()

print ('Age spread: ')
# Plotting to undertanding the relation ship between age and survival rate:



face_age = sns.FacetGrid(train, hue='Survived', size=3,aspect=3)

face_age.map(sns.kdeplot, 'Age', shade=True)

face_age.set(xlim=(0, train.Age.max()))

face_age.add_legend()



plt.subplots(1,1, figsize=(10,4))

average_age = train[['Age', 'Survived']].groupby(['Age'], as_index=False).mean()

sns.barplot('Age', 'Survived', data=average_age)

plt.xticks(rotation=90)

print ('Age survival relation: ')
# combing the parents children count and siblings count for each person to get the family size:

train['Family'] = train.Parch + train.SibSp

test['Family'] = test.Parch + test.SibSp



#making the values to boolean, i.e. if no family member family is 0 else 1.

train.Family.loc[train.Family > 0] = 1

train.Family.loc[train.Family == 0] = 0



test.Family.loc[test.Family > 0] = 1

test.Family.loc[test.Family == 0] = 0



#Drop the original features since they are not required any more:

train.drop(['Parch', 'SibSp'], inplace=True, axis=1)

test.drop(['Parch', 'SibSp'], inplace=True, axis=1)



#relation between family and survival rate:

sns.countplot('Family', hue='Survived', data=train)



print ('Family survival rate: ')
#There is a chance that more children survived the disaster, therefore lets put three categories

#i.e. male, female and childre with age less then 17



def check_child(age_gender):

    age, sex = age_gender

    return 'child' if age < 17 else sex



train['Person'] = train[['Age', 'Sex']].apply(check_child, axis=1)

test['Person'] = test[['Age', 'Sex']].apply(check_child, axis=1)



#creating dummies the new person feature for our model

train = pd.concat([train, pd.get_dummies(train.Person, prefix='person')], axis=1)

test = pd.concat([test, pd.get_dummies(test.Person, prefix='person')], axis=1)



train.head()
# Check the spread of each person and their survival rate:



_, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(10,12))



#spread

sns.countplot('Person', data=train, ax=ax1)



#survival

sns.countplot('Person', hue='Survived', data=train, ax=ax2)



#mean-survival

person_survival = train[['Person', 'Survived']].groupby(['Person'], as_index=False).mean()

sns.barplot('Person', 'Survived', data=person_survival, ax=ax3)



#Drop the original features, they are no more needed.

train.drop(['Sex', 'Person'], inplace=True, axis=1)

test.drop(['Sex', 'Person'], inplace=True, axis=1)
# There might be strong realtion between survival rate and the traveling class of a person.

# plot

sns.countplot('Pclass', hue='Survived', data=train)



#create dummy variable for out model

train = pd.concat([train, pd.get_dummies(train.Pclass, prefix='pclass')], axis=1)

test = pd.concat([test, pd.get_dummies(test.Pclass, prefix='pclass')], axis=1)



#drop original feature:

train.drop('Pclass', inplace=True, axis=1)

test.drop('Pclass', inplace=True, axis=1)
#seperate the features and target:

X = train.drop('Survived', axis=1)

y = train.Survived

#split the data to evalute the model:

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, random_state=4)
#Logit regression will simply classify the data set using l2 regularization:

logreg = LogisticRegression(C=1, penalty='l2').fit(X_train, y_train)

logreg.score(X_train, y_train), logreg.score(X_test, y_test)
#The coefficient found for each feature, this slightly tells 

#which feature tells more about survival and death:

plt.plot(logreg.coef_.T, 'o')

plt.xticks(range(X.shape[1]), X.columns, rotation=90)

print ('Coefficients found using Logistic regression: ')
#Decision trees will give more accurate result since it calculates the score from eah feature multiple times

#Pre-prunning has been applied here to avoid over fitting

dtree = DecisionTreeClassifier(random_state=0, max_depth=3).fit(X_train, y_train)

dtree.score(X_train, y_train), dtree.score(X_test, y_test)
#Check out the feature importance taken into account by Decision tree.



plt.plot(dtree.feature_importances_, 'o')

plt.xticks(range(X.shape[1]), X.columns, rotation=90)

print ('Feature Importance in Decision trees: ')
# Using ensemble techinque will cause imporvment in result. Here the max depth is again changed.



rf = RandomForestClassifier(random_state=0, max_depth=4).fit(X_train, y_train)

rf.score(X_train, y_train), rf.score(X_test, y_test)
# Feature importance by random forest, here more features are taken into account.



plt.plot(rf.feature_importances_, 'o')

plt.xticks(range(X.shape[1]), X_train.columns, rotation=90)

print ('Feature Importance in Random Forest: ')
#This is will give more accuracy since it learn from the mistakes of the previous trees.

#Pre-prunning is used here to avoid over fitting and for better accuracy.



gb = GradientBoostingClassifier(learning_rate=0.1, random_state=0, max_depth=1).fit(X_train, y_train)

gb.score(X_train, y_train), gb.score(X_test, y_test)
#Feature importance for the gb



plt.plot(gb.feature_importances_, 'o')

plt.xticks(range(X.shape[1]), X_train.columns, rotation =90)

print ('Feature Importance in Gradient Booster: ')
pred = test.drop('PassengerId', axis=1)

prediction = gb.predict(pred)
output = pd.DataFrame({

        'PassengerId': test.PassengerId,

        'Survived': prediction

    })

output.to_csv('result.csv', index=False)