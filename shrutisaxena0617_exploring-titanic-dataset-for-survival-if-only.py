# Load python packages

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#import matplotlib as plt

import matplotlib.pyplot as plt 

import seaborn as sns

from sklearn.naive_bayes import GaussianNB

from sklearn import svm

from sklearn import tree

from sklearn.metrics import accuracy_score

from sklearn import preprocessing

from sklearn.model_selection import train_test_split



# Load test and training data

test_df = pd.read_csv('../input/test.csv')

train_df = pd.read_csv('../input/train.csv')
# Display test data

test_df.head()
# Display training data

train_df.head()
# Volume of test data is 418 rows

test_df.count()
# Volume of training data is 891

train_df.count()
# Computing ratio of training and test data sets

len(train_df) / (len(test_df) + len(train_df))
# Check if any duplicate rows are present in training data. 

# Checking with PassengerId as it should be unique

train_df['PassengerId'].duplicated().any()
# Check data type of all columns in training data

train_df.dtypes
# Handling missing values in 'Age'. 

train_df['Age'].isnull().sum()
# Proportion of age data with missing values

train_df['Age'].isnull().sum() / len(train_df)
train_df['Age'].hist(bins = 18, color = 'darkturquoise')
# Handling missing values in 'Cabin'. 

train_df['Cabin'].isnull().sum()
# Proportion of Cabin data with missing values

train_df['Cabin'].isnull().sum() / len(train_df)
# Handling missing values in 'Emabarked'. 

train_df['Embarked'].isnull().sum()
# Proportion of Embarked data with missing values

train_df['Embarked'].isnull().sum() / len(train_df)
# Embarked is a categorical variable. Let's see its distribution

#train_df.plot(kind='bar', title ='Embarked classes',figsize=(15,10),legend=True, fontsize=12)

train_df.Embarked.value_counts().plot.barh(color = 'darkturquoise')
# Handling missing values in 'Age'. 

test_df['Age'].isnull().sum()
# Proportion of age data with missing values

test_df['Age'].isnull().sum() / len(test_df)
test_df['Age'].hist(bins = 18, color = 'salmon')
# Handling missing values in 'Fare'. 

test_df['Fare'].isnull().sum()
# Proportion of fare data with missing values

test_df['Fare'].isnull().sum() / len(test_df)
test_df['Fare'].hist(bins = 18, color = 'salmon')
test_df[pd.isnull(test_df['Fare'])]
# Handling missing values in 'Cabin'. 

test_df['Cabin'].isnull().sum()
# Proportion of cabin data with missing values

test_df['Cabin'].isnull().sum() / len(test_df)
train_df_bkp = train_df

train_df_bkp.head()
train_df['Age'].fillna(train_df['Age'].median(skipna = True), inplace = True)

train_df['Age'].isnull().sum()
train_df.drop('Cabin', axis = 1, inplace = True)

train_df.head()
train_df['Embarked'].fillna('S', inplace = True)

train_df['Embarked'].isnull().sum()
# Drop 'Name', 'PassengerId' and 'Ticket'

# Ticket would have been useful in case class / cabin / fare / Embarked were not included in data

train_df.drop('PassengerId', axis=1, inplace=True)

train_df.drop('Name', axis=1, inplace=True)

train_df.drop('Ticket', axis=1, inplace=True)

train_df.head()
train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1

train_df['TraveledAlone'] = (train_df['FamilySize'] == 1).astype(int)

train_df.drop('FamilySize', axis=1, inplace=True)

train_df.drop('SibSp', axis=1, inplace=True)

train_df.drop('Parch', axis=1, inplace=True)

train_df.head()
# Creating categorical variables for 'Pclass', 'Sex', and 'Embarked'

train_df_final = pd.get_dummies(train_df, columns=["Pclass", "Sex", "Embarked"])

train_df_final.drop('Sex_female', axis = 1, inplace = True)

train_df_final.head()
test_df_bkp = test_df

passengerId = test_df_bkp['PassengerId']

test_df_bkp.head()
test_df['Age'].fillna(test_df['Age'].median(skipna = True), inplace = True)

test_df['Age'].isnull().sum()
test_df['Fare'].fillna(test_df[(test_df['Pclass'] == 3)].mean(0)['Fare'], inplace = True)

#test_df[(test_df['Pclass'] == 3)].mean(0)['Fare']

test_df['Fare'].isnull().sum()
test_df.drop('Cabin', axis = 1, inplace = True)

test_df.head()
# Drop 'Name', 'PassengerId' and 'Ticket'

test_df.drop('PassengerId', axis=1, inplace=True)

test_df.drop('Name', axis=1, inplace=True)

test_df.drop('Ticket', axis=1, inplace=True)

test_df.head()
test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch'] + 1

test_df['TraveledAlone'] = (test_df['FamilySize'] == 1).astype(int)

test_df.drop('FamilySize', axis=1, inplace=True)

test_df.drop('SibSp', axis=1, inplace=True)

test_df.drop('Parch', axis=1, inplace=True)

test_df.head()
# Creating categorical variables for 'Pclass', 'Sex', and 'Embarked'

test_df_final = pd.get_dummies(test_df, columns=["Pclass", "Sex", "Embarked"])

test_df_final.drop('Sex_female', axis = 1, inplace = True)

test_df_final.head()
train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean()
train_df[["Pclass", "Survived"]].groupby(['Pclass'], as_index=False).mean()
train_df[["Embarked", "Survived"]].groupby(['Embarked'], as_index=False).mean()
train_df[["TraveledAlone", "Survived"]].groupby(['TraveledAlone'], as_index=False).mean()
plt.figure(figsize=(10,6))

sns.kdeplot(train_df["Fare"][train_df.Survived == 1], color="darkturquoise", shade=True)

sns.kdeplot(train_df["Fare"][train_df.Survived == 0], color="salmon", shade=True)

plt.xlim(-20,300)

plt.legend(['Survived', 'Not survived'])

plt.title('Density Plot of Fare v/s Survived')

plt.show()
plt.figure(figsize=(10,6))

sns.kdeplot(train_df["Age"][train_df.Survived == 1], color="darkturquoise", shade=True)

sns.kdeplot(train_df["Age"][train_df.Survived == 0], color="salmon", shade=True)

#plt.xlim(-20,300)

plt.legend(['Survived', 'Not survived'])

plt.title('Density Plot of Age v/s Survived')

plt.show()
#train_df.head()

#train_label = train_df.iloc[:, 0]

#train_features = train_df[['Sex', 'Pclass', 'Fare', 'Age', 'TraveledAlone', 'Embarked']]

#test_features = test_df[['Sex', 'Pclass', 'Fare', 'Age', 'TraveledAlone', 'Embarked']]
train_label = train_df_final.iloc[:, 0]

train_features = train_df_final[['Age', 'Fare', 'TraveledAlone', 'Pclass_1', 'Pclass_2', 'Sex_male', 'Embarked_C']]

test_features = test_df_final[['Age', 'Fare', 'TraveledAlone', 'Pclass_1', 'Pclass_2', 'Sex_male', 'Embarked_C']]
clf = GaussianNB() 

clf = clf.fit(train_features, train_label)

clf.score(train_features, train_label)
train, test = train_test_split(train_df_final, test_size=0.2)
validation_train_features = train[['Age', 'Fare', 'TraveledAlone', 'Pclass_1', 'Pclass_2', 'Sex_male', 'Embarked_C']]

validation_train_label = train['Survived']

validation_test_features = test[['Age', 'Fare', 'TraveledAlone', 'Pclass_1', 'Pclass_2', 'Sex_male', 'Embarked_C']]

validation_test_label = test['Survived']
clf_test = GaussianNB() 

clf_test = clf_test.fit(validation_train_features, validation_train_label)

#clf_test.score(validation_train_features, validation_train_label)

pred = clf_test.predict(validation_test_features)

accuracy = accuracy_score(pred, validation_test_label)

print(accuracy)
clf = svm.SVC(kernel = 'rbf', C = 10) 

clf = clf.fit(train_features, train_label)

clf.score(train_features, train_label)
train, test = train_test_split(train_df_final, test_size=0.2)

validation_train_features = train[['Age', 'Fare', 'TraveledAlone', 'Pclass_1', 'Pclass_2', 'Sex_male', 'Embarked_C']]

validation_train_label = train['Survived']

validation_test_features = test[['Age', 'Fare', 'TraveledAlone', 'Pclass_1', 'Pclass_2', 'Sex_male', 'Embarked_C']]

validation_test_label = test['Survived']

clf_test = svm.SVC(kernel = 'rbf', C = 10) 

clf_test = clf_test.fit(validation_train_features, validation_train_label)

#clf_test.score(validation_train_features, validation_train_label)

pred = clf_test.predict(validation_test_features)

accuracy = accuracy_score(pred, validation_test_label)

print(accuracy)
clf = tree.DecisionTreeClassifier(min_samples_split = 10) 

clf = clf.fit(train_features, train_label)

clf.score(train_features, train_label)
train, test = train_test_split(train_df_final, test_size=0.2)

validation_train_features = train[['Age', 'Fare', 'TraveledAlone', 'Pclass_1', 'Pclass_2', 'Sex_male', 'Embarked_C']]

validation_train_label = train['Survived']

validation_test_features = test[['Age', 'Fare', 'TraveledAlone', 'Pclass_1', 'Pclass_2', 'Sex_male', 'Embarked_C']]

validation_test_label = test['Survived']

clf_test = tree.DecisionTreeClassifier(min_samples_split = 10) 

clf_test = clf_test.fit(validation_train_features, validation_train_label)

#clf_test.score(validation_train_features, validation_train_label)

pred = clf_test.predict(validation_test_features)

accuracy = accuracy_score(pred, validation_test_label)

print(accuracy)
clf = tree.DecisionTreeClassifier(min_samples_split = 10) 

clf = clf.fit(train_features, train_label)

preds = clf.predict(test_features)
submission = pd.DataFrame({

        'PassengerId': passengerId,

        "Survived": preds

    })

submission.to_csv('results.csv', header = True, index=False)