# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# Any results you write to the current directory are saved as output.
input_path = "../input/titanic/"

df_train = pd.read_csv(input_path+'train.csv')

df_test = pd.read_csv(input_path+'test.csv')
df_train.head()
average_age = np.sum(df_train['Age']/len(df_train['Age']))

df_train.loc[df_train["Age"].isnull(),"Age"] = average_age

df_test.loc[df_test["Age"].isnull(),"Age"] = average_age



average_fare = np.sum(df_train['Fare']/len(df_train['Fare']))

df_train.loc[df_train["Fare"].isnull(),"Fare"] = average_fare 

df_test.loc[df_test["Fare"].isnull(),"Fare"] = average_fare



df_train.loc[df_train["Parch"].isnull(),"Parch"] = 0

df_test.loc[df_test["Parch"].isnull(),"Parch"] = 0



df_train.loc[df_train["Embarked"].isnull(),"Embarked"] = "S"

df_test.loc[df_test["Embarked"].isnull(),"Embarked"] = "S"



df_train["CabinIsNull"] = np.where(df_train["Cabin"].isnull(), 1, 0)

df_test["CabinIsNull"] = np.where(df_test["Cabin"].isnull(), 1, 0)

df_train.loc[df_train['Cabin'].isnull(),'Cabin'] = 'N'

df_test.loc[df_test['Cabin'].isnull(),'Cabin'] = 'N'
df_train.head()
df_train.isna().sum()
sns.barplot(x='Embarked', y='Survived', data=df_train);
#création d'une colone pour les femmes jeunes et riches

df_train['YoungAndRichFemale'] = 0

df_train.loc[(df_train['Sex'] == 'female') & (df_train['Age'] <= 30) & (df_train['Pclass'] <= 2), 'YoungAndRichFemale'] = 1

df_train['YoungAndRichMale'] = 0

df_train.loc[(df_train['Sex'] == 'male') & (df_train['Age'] <= 30) & (df_train['Pclass'] <= 2), 'YoungAndRichMale'] = 1
#création d'une colone pour les hommes jeunes et riches

df_test['YoungAndRichFemale'] = 0

df_test.loc[(df_test['Sex'] == 'female') & (df_test['Age'] <= 30) & (df_test['Pclass'] <= 2), 'YoungAndRichFemale'] = 1

df_test['YoungAndRichMale'] = 0

df_test.loc[(df_test['Sex'] == 'male') & (df_test['Age'] <= 30) & (df_test['Pclass'] <= 2), 'YoungAndRichMale'] = 1
#comparatif des femmes jeune et riche (1) ou non (0) qui ont survécut

sns.barplot(x='YoungAndRichFemale', y='Survived', data=df_train);
#comparatif des hommes jeunes et riches (1) ou non (0) qui ont survécut

sns.barplot(x='YoungAndRichMale', y='Survived', data=df_train);
#affiche le nombre de mineur présent sur le bateau

nb_minor = df_train['PassengerId'].loc[df_train['Age'] <= 18].size

nb_minor
#création d'une colone pour les mineurs avec parents

df_train['MinorWithParch'] = 0

df_train.loc[(df_train['Age'] <= 18) & (df_train['Parch'] != 0), 'MinorWithParch'] = 1

df_test['MinorWithParch'] = 0

df_test.loc[(df_train['Age'] <= 18) & (df_test['Parch'] != 0), 'MinorWithParch'] = 1
#comparatif des mineurs qui avait des parents (1) ou non (0) qui ont survécut

sns.barplot(x='MinorWithParch', y='Survived', data=df_train);
#comparatif du nombre de survivant en fonction de la première lettre de la cabine (N représnte les cabine NAN)

sns.barplot(x = df_test['Cabin'].str[0], y = 'Survived', data = df_train);
#création des colones en fonction du nombres de survivant par emplacement des cabines pour df_train

letter_cabin = df_train['FirstLetterCabin'] = df_train.Cabin.str.slice(0,1)

df_train['TheBestCabin'] = 0

df_train['CabinIsGood'] = 0

df_train['CabinIsMedium'] = 0

df_train['CabinIsBad'] = 0

df_train['TheWorstCabin'] = 0

df_train.loc[(letter_cabin  == 'F'), 'TheBestCabin'] = 1

df_train.loc[(letter_cabin  == 'B') | (letter_cabin  == 'A') | (letter_cabin  == 'C'), 'CabinIsGood'] = 1

df_train.loc[(letter_cabin  == 'N') | (letter_cabin  == 'D'), 'CabinIsMedium'] = 1

df_train.loc[(letter_cabin  == 'E'), 'CabinIsBad'] = 1

df_train.loc[(letter_cabin  == 'G'), 'TheWorstCabin'] = 1
#création des colones en fonction du nombres de survivant par emplacement des cabines pour df_test

letter_cabin = df_test['FirstLetterCabin'] = df_test.Cabin.str.slice(0,1)

df_test['TheBestCabin'] = 0

df_test['CabinIsGood'] = 0

df_test['CabinIsMedium'] = 0

df_test['CabinIsBad'] = 0

df_test['ProbablyDead'] = 0

df_test['TheWorstCabin'] = 0

df_test.loc[(letter_cabin  == 'F'), 'TheBestCabin'] = 1

df_test.loc[(letter_cabin  == 'B') | (letter_cabin  == 'A') | (letter_cabin  == 'C'), 'CabinIsGood'] = 1

df_test.loc[(letter_cabin  == 'N') | (letter_cabin  == 'D'), 'CabinIsMedium'] = 1

df_test.loc[(letter_cabin  == 'E'), 'CabinIsBad'] = 1

df_test.loc[(letter_cabin  == 'G'), 'TheWorstCabin'] = 1
df_train.isna().sum()
NUM_FEATURES = ['Age','Fare', 'YoungAndRichFemale', 'YoungAndRichFemale', 'TheBestCabin', 'CabinIsGood', 'CabinIsMedium', 'CabinIsBad', 'TheWorstCabin', 'CabinIsNull', 'MinorWithParch']

CAT_FEATURES = ['Sex', 'Embarked', 'Pclass']

for c in CAT_FEATURES:

    df_train[c] = df_train[c].astype('category').cat.codes
for c in CAT_FEATURES:

    df_test[c] = df_test[c].astype('category').cat.codes
# On prépare le dataset pour le modèle d'integration

X_train = df_train.loc[:, NUM_FEATURES]

X_train_cat = df_train.loc[:, CAT_FEATURES]

y_train = df_train['Survived']
X_test = df_test.loc[:, NUM_FEATURES]

X_test_cat = df_test.loc[:, CAT_FEATURES]
pd.concat([X_train, X_train_cat], axis = 1)

pd.concat([X_test, X_test_cat], axis = 1)
#affiche le nombre total de passagers du titanic (passagers / hommes / femmes)

sum_passenger = df_train['PassengerId'].size

sum_passenger_m =  df_train['Sex'].loc[df_train['Sex'] == 1].size

sum_passenger_f =  df_train['Sex'].loc[df_train['Sex'] == 0].size

sum_passenger
nbr = df_train[['Sex','Embarked']].groupby('Sex').count().sort_values(by='Embarked', ascending=False)

nbr.reset_index(0, inplace=True)

nbr.head()
#affiche le nombres total de passager sur le Titanic par genre (0 pour les hommes et 1 pour els femmes)

sns.barplot(x=nbr['Sex'], y=nbr['Embarked'], palette="Reds_r")

plt.xlabel('Genre des passagers', fontsize=15, color='#c0392b')

plt.ylabel('Nombre de personnes', fontsize=15, color='#c0392b')

plt.title("Nombre total de passagers (H/F) sur le Titanic", fontsize=18, color='#e74c3c')

plt.tight_layout()
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
train_acc = accuracy_score(y_train, rf.predict(X_train))

print(train_acc)
rf.predict(X_test)
# cellule pour stocker le résultat du modèle en appelant son modèle rf et son set de test X_test

submission = pd.DataFrame({'PassengerId' : df_test['PassengerId'],

                           'Survived' : rf.predict(X_test)})

submission.to_csv('submission.csv', index=False)