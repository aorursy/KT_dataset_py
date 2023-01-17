import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')



from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, LinearRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
# Train-data

print(train_df.shape)

train_df.head()
# Test-data

print(test_df.shape)

test_df.head()
train_df.describe()
print("Train Data")

train_df.info()

print("------------------------------------------")

print("Test Data")

test_df.info()
# missing values

train_df[(train_df['Embarked']).isnull()==True]
#  Numbers in each Embark(S,C,Q) categories

fig, axis1 = plt.subplots(1, figsize=(15,5))

sns.countplot(x='Embarked', data=train_df, ax=axis1)
train_df.groupby('Embarked').count() # The number of  'S' is highest
# S,C,Q × Survived(0,1)

fig, axis1 = plt.subplots(1, figsize=(15,5))

sns.countplot(x='Survived', hue='Embarked', data=train_df, order=[0,1], ax=axis1)  
sns.factorplot('Embarked', 'Survived', data=train_df,  aspect=3)
train_df.groupby('Embarked').mean()['Survived'] # Mean of 'Survived' in each categories
# Fill missing values

train_df['Embarked'] = train_df['Embarked'].fillna('S')
# Make dummy variables of 'Embarked'



# Train-data

embark_dummy_train = pd.get_dummies(train_df['Embarked'])

embark_dummy_train.drop(['S'],axis=1, inplace=True)

# Test-dta

embark_dummy_test = pd.get_dummies(test_df['Embarked'])

embark_dummy_test.drop(['S'],axis=1, inplace=True)



train_df = train_df.join(embark_dummy_train)

test_df = test_df.join(embark_dummy_test)

train_df.drop(['Embarked'], axis=1,inplace=True)

test_df.drop(['Embarked'], axis=1, inplace=True)
train_df.head()
# missing value

test_df[(test_df['Fare']).isnull()==True]
# Fill missing value by mean of Fare

test_df['Fare'].fillna(test_df['Fare'].mean(), inplace=True)
# Histgram of Fare



# fig, axis1 = plt.subplots(1, figsize=(15,5))

# sns.distplot(train_df['Fare'], ax=axis1)



train_df['Fare'].plot(kind='hist', figsize=(15,5), bins=100, xlim=(0,300))
fare_not_survived = train_df['Fare'][train_df['Survived']==0]

fare_survived = train_df['Fare'][train_df['Survived']==1]



print("Survived: Mean %f, SD %f" % (fare_survived.mean(),fare_survived.mean()))

print("Not Survived: Mean %f, SD %f" % (fare_not_survived.mean(),fare_not_survived.std()))
average_age_train = train_df['Age'].mean()

std_age_train = train_df['Age'].std()

count_nan_age_train = train_df['Age'].isnull().sum()



average_age_test = test_df['Age'].mean()

std_age_test = test_df['Age'].std()

count_nan_age_test = test_df['Age'].isnull().sum()



print("Train-data")

print("Mean: %.2f, SD: %.2f, missing: %i"% (average_age_train, std_age_train, count_nan_age_train))



print("Test-data")

print("Mean: %.2f, SD: %.2f, missing: %i" % (average_age_test, std_age_test, count_nan_age_test))
# random values from Mean-SD to Mean+SD

rand_train = np.random.randint(average_age_train-std_age_train, average_age_train+std_age_train, size=count_nan_age_train)

rand_test = np.random.randint(average_age_test-std_age_test, average_age_test+std_age_test, size=count_nan_age_test)
# Fill missing values

train_df['Age'][np.isnan(train_df['Age'])] = rand_train

test_df['Age'][np.isnan(test_df['Age'])] = rand_test



train_df['Age'] = train_df['Age'].astype(int)

test_df['Age'] = test_df['Age'].astype(int)
# Histgram of Age

facet = sns.FacetGrid(data=train_df, hue='Survived', aspect=4)

facet.map(sns.distplot, 'Age')

facet.set(xlim=(0, train_df['Age'].max()))

facet.add_legend()
age_not_survived = train_df['Age'][train_df['Survived']==0]

age_survived = train_df['Age'][train_df['Survived']==1]



print("Survived: Mean %f, SD %f" % (age_survived.mean(),age_survived.mean()))

print("Not Survived: Mean %f, SD %f" % (age_not_survived.mean(),age_not_survived.std()))
def get_person(passenger):

    age, sex = passenger

    return 'child' if age < 16 else sex



train_df['Person'] = train_df[['Age', 'Sex']].apply(get_person, axis=1)

test_df['Person'] = test_df[['Age', 'Sex']].apply(get_person, axis=1)



train_df.drop(['Sex'], axis=1, inplace=True)

test_df.drop(['Sex'], axis=1, inplace=True)
# Histgram of Person

flg, axis1= plt.subplots(1,  figsize=(15,5))

sns.countplot(x='Person', data=train_df, ax=axis1)
# Mean of  Survived

flg, axis1= plt.subplots(1,  figsize=(15,5))

person_perc = train_df[['Person', 'Survived']].groupby('Person', as_index=False).mean()

sns.barplot(x='Person', y='Survived', data=person_perc, ax=axis1, order=['male', 'female', 'child'])
train_df[['Person', 'Survived']].groupby('Person', as_index=False).mean()
# Make dummy variables

person_dummy_train = pd.get_dummies(train_df['Person'])

person_dummy_train.columns = ['Child', 'Female', 'Male']

person_dummy_train.drop(['Male'], axis=1, inplace=True) 



person_dummy_test = pd.get_dummies(test_df['Person'])

person_dummy_test.columns = ['Child', 'Female', 'Male']

person_dummy_test.drop(['Male'], axis=1, inplace=True)



train_df = train_df.join(person_dummy_train)

test_df = test_df.join(person_dummy_test)



train_df.drop('Person', axis=1, inplace=True)

test_df.drop('Person', axis=1, inplace=True)
train_df.head()
count_nan_age_train = train_df['Cabin'].isnull().sum()

count_nan_age_test = test_df['Cabin'].isnull().sum()



print("Train-data")

print("count: %i, missing values: %i"% (train_df.shape[0], count_nan_age_train))

print("Test-data")

print("count: %i, missing values: %i" % (test_df.shape[0], count_nan_age_test))
train_df.drop('Cabin', axis=1, inplace=True)

test_df.drop('Cabin', axis=1, inplace=True)
parch_not_survived = train_df['Parch'][train_df['Survived']==0]

parch_survived = train_df['Parch'][train_df['Survived']==1]



print("Survived: Mean %f, SD %f" % (parch_survived.mean(),parch_survived.mean()))

print("Not-Survived: Mean %f, SD %f" % (parch_not_survived.mean(),parch_not_survived.std()))
sibsp_not_survived = train_df['SibSp'][train_df['Survived']==0]

sibsp_survived = train_df['SibSp'][train_df['Survived']==1]



print("Survived: Mean %f, SD %f" % (sibsp_survived.mean(),sibsp_survived.mean()))

print("Not-Survived: Mean %f, SD %f" % (sibsp_not_survived.mean(),sibsp_not_survived.std()))
sns.factorplot(x='SibSp', y='Survived', data=train_df,  aspect=3)
sns.factorplot(x='Parch', y='Survived', data=train_df,  aspect=3)
# Make 'Family'  category

train_df['Family'] = train_df['Parch']+train_df['SibSp']

train_df['Family'].loc[train_df['Family']>0] = 1

train_df['Family'].loc[train_df['Family']==0] = 0



test_df['Family'] = test_df['Parch']+test_df['SibSp']

test_df['Family'].loc[test_df['Family']>0] = 1

test_df['Family'].loc[test_df['Family']==0] = 0



train_df = train_df.drop(['Parch', 'SibSp'], axis=1)

test_df = test_df.drop(['Parch', 'SibSp'], axis=1)
flg, (axis1, axis2) = plt.subplots(1,2, sharex=True, figsize=(15,5))



# Count

sns.countplot(x='Family', data=train_df, order=[1,0], ax=axis1)



# Mean of Survived

family_perc = train_df[['Family', 'Survived']].groupby(['Family'], as_index=False).mean()

sns.barplot(x='Family', y='Survived', data=family_perc, order=[1,0], ax=axis2)



axis1.set_xticklabels(['2 more peaple', 'Alone'], rotation=0)
sns.factorplot(x='Pclass', y='Survived', data=train_df, order=[1,2,3],  aspect=3)
train_df[['Pclass', 'Survived']].groupby('Pclass', as_index=False).mean()
# Make dummy variable

pclass_dummy_train = pd.get_dummies(train_df['Pclass'])

pclass_dummy_train.columns = ['Class_1','Class_2','Class_3']

pclass_dummy_train.drop('Class_3', axis=1, inplace=True)



pclass_dummy_test = pd.get_dummies(test_df['Pclass'])

pclass_dummy_test.columns = ['Class_1','Class_2','Class_3']

pclass_dummy_test.drop('Class_3', axis=1, inplace=True)



train_df.drop('Pclass', axis=1, inplace=True)

test_df.drop('Pclass', axis=1, inplace=True)



train_df = train_df.join(pclass_dummy_train)

test_df = test_df.join(pclass_dummy_test)
train_df.head()
# Delete variables

train_df = train_df.drop(['PassengerId', 'Name', 'Ticket'], axis=1)

test_df = test_df.drop(['Name', 'Ticket'], axis=1)
X_train = train_df.drop('Survived', axis=1)

Y_train = train_df['Survived']

X_test = test_df.drop('PassengerId', axis=1).copy()
X_test.head()
logreg = LogisticRegression()

logreg.fit(X_train, Y_train)  

Y_pred = logreg.predict(X_test)

logreg.score(X_train, Y_train)  
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)
svc = SVC()

svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)

svc.score(X_train, Y_train)
knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

knn.score(X_train, Y_train)
gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

Y_pred = gaussian.predict(X_test)

gaussian.score(X_train, Y_train)
logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test) 



submission = pd.DataFrame({

    'PassengerId' : test_df['PassengerId'],

    'Survived' : Y_pred

})



# submission.to_csv('titanic_logreg_1119.csv', index=False)
random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)



submission = pd.DataFrame({

    'PassengerId' : test_df['PassengerId'],

    'Survived' : Y_pred

})



# submission.to_csv('titanic_randomforest_1119.csv', index=False)
svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)



submission = pd.DataFrame({

    'PassengerId' : test_df['PassengerId'],

    'Survived' : Y_pred

})



# submission.to_csv('titanic_svm_1119.csv', index=False)
knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)



submission = pd.DataFrame({

    'PassengerId' : test_df['PassengerId'],

    'Survived' : Y_pred

})



# submission.to_csv('titanic_knn_1119.csv', index=False)
gaussian.fit(X_train, Y_train)

Y_pred = gaussian.predict(X_test)



submission = pd.DataFrame({

    'PassengerId' : test_df['PassengerId'],

    'Survived' : Y_pred

})



# submission.to_csv('titanic_gaussian_1119.csv', index=False)