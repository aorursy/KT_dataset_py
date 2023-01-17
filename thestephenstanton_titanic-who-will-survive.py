import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
titanic_train = pd.read_csv("../input/train.csv")

titanic_train.head(5)
titanic_test = pd.read_csv("../input/test.csv")
titanic = titanic_train.copy()
titanic.info()
titanic.Survived.value_counts(normalize=True)
titanic.Pclass.value_counts(normalize=True)
titanic[titanic.Survived == 1].Pclass.value_counts() / titanic.Pclass.value_counts()
titanic.Name.sample(5, random_state=1)
name_prefixs = titanic.Name.str.extract(',\s?(.*?)\.')

name_prefixs.value_counts()
print('Survived:')

for prefix in name_prefixs.unique():

    temp = titanic[titanic.Name.str.contains('%s\.' % (prefix))]

    print(prefix, len(temp[temp.Survived == 1]) / len(temp))
titanic.Sex.value_counts(normalize=True)
titanic[titanic.Survived == 1].Sex.value_counts() / titanic.Sex.value_counts()
print('Master:')

print(titanic[titanic.Name.str.contains('Master\.')].Sex.value_counts())

print('\n')

print('Dr:')

print(titanic[titanic.Name.str.contains('Dr\.')].Sex.value_counts())
titanic[titanic.Name.str.contains('Dr\.')]
print('Mrs:')

print(titanic[titanic.Name.str.contains('Mrs\.')].Survived.value_counts())

print('Total:', len(titanic[titanic.Name.str.contains('Mrs\.')]))

print('\n')

print('Miss:')

print(titanic[titanic.Name.str.contains('Miss\.')].Survived.value_counts())

print('Total:', len(titanic[titanic.Name.str.contains('Miss\.')]))
titanic.Age.describe()
counts = titanic.groupby(['Age', 'Survived']).Age.count().unstack()

counts.plot(kind='bar', stacked=True, figsize=(15,10))
bins = np.arange(0, 100, 10)

labels = list(map(str, bins[:-1]))

labels = [b + 's' for b in labels]

titanic_age_bins = titanic.copy()



titanic_age_bins['AgeBin'] = pd.cut(titanic_age_bins.Age, bins=bins, labels=labels)

titanic_age_bins.head()
counts = titanic_age_bins.groupby(['AgeBin', 'Survived']).AgeBin.count().unstack()

counts.plot(kind='bar', stacked=True, figsize=(15,10))
for age_bin in [str(b) + 's' for b in np.arange(0, 80, 10)]:

    print(age_bin, 'survival:')

    temp_bin = titanic_age_bins[titanic_age_bins.AgeBin == age_bin]

    print(temp_bin[temp_bin.Survived == 1].Pclass.value_counts())

    print('Survived / Total in Bin:', len(temp_bin[temp_bin.Survived == 1]) / len(temp_bin))

    print('\n')
titanic.SibSp.value_counts()
titanic[titanic.Survived == 1].SibSp.value_counts() / titanic.SibSp.value_counts()
titanic[titanic.SibSp >= 4].sample(10, random_state = 42)
titanic[titanic.SibSp > 2].Pclass.value_counts()
titanic[titanic.Survived == 1][titanic.SibSp > 2].Pclass.value_counts() / titanic[titanic.SibSp > 2].Pclass.value_counts()
titanic.Parch.value_counts()
titanic[titanic.Survived == 1].Parch.value_counts() / titanic.Parch.value_counts()
titanic.Fare.describe()
print('1st class')

print(titanic[titanic.Pclass == 1].Fare.describe())

print('\n')

print('2nd class')

print(titanic[titanic.Pclass == 2].Fare.describe())

print('\n')

print('3rd class')

print(titanic[titanic.Pclass == 3].Fare.describe())
survived = titanic[titanic.Survived == 1]



print('Survival rate for all of 1st class')

print(len(survived[titanic.Pclass == 1]) / len(titanic[titanic.Pclass == 1]))

print('Survival rate for of 1st class who paid over mean')

print(len(survived[titanic.Pclass == 1][titanic.Fare > titanic.Fare.mean()]) / len(titanic[titanic.Pclass == 1]))



print('\n')



print('Survival rate for all of 3rd class')

print(len(survived[titanic.Pclass == 3]) / len(titanic[titanic.Pclass == 3]))

print('Survival rate for of 3rd class who paid under mean')

print(len(survived[titanic.Pclass == 3][titanic.Fare < titanic.Fare.mean()]) / len(titanic[titanic.Pclass == 3]))
titanic[titanic.Survived == 1].Embarked.value_counts() / titanic.Embarked.value_counts()
titanic.Embarked.value_counts()
titanic[titanic.Embarked == 'C'].Pclass.value_counts()
for port in ['S', 'C', 'Q']:

    print('Port', port)

    print(titanic[titanic.Embarked == port][titanic.Pclass == 3].Survived.value_counts(normalize = True))
# Sex

one_hot_sex = pd.get_dummies(titanic.Sex) # lol 



# Age

one_hot_age_bins = pd.get_dummies(titanic_age_bins.AgeBin)



# Embarked

titanic_embarked_filled = titanic.Embarked.fillna(titanic.Embarked.value_counts().idxmax())

one_hot_embarked = pd.get_dummies(titanic_embarked_filled)



# Apply all to titanic

one_hot_titanic = pd.concat([titanic, one_hot_sex, one_hot_age_bins, one_hot_embarked], axis=1)

one_hot_titanic.info()
correlation_matrix = one_hot_titanic.corr()

correlation_matrix.Survived.sort_values(ascending=False)
correlation_matrix = titanic.corr()

correlation_matrix.Pclass.sort_values(ascending=False)
pd.concat([titanic, pd.get_dummies(titanic.Sex)], axis=1).corr().Survived.sort_values(ascending=False)