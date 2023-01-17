# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

input_path = "../input/titanic/"

df_train = pd.read_csv(input_path + 'train.csv')

df_test = pd.read_csv(input_path + 'test.csv')

df_gen = pd.read_csv(input_path + 'gender_submission.csv')
# Vérifi l'existance des valeurs null

df_train.isna().sum()

df_test.isna().sum()
int(df_train['Age'].mean())
#on créé une nouvelle colonne pour stocker en boolean si l'age est renseigné ou pas

df_train['age_is_null'] = np.where(df_train['Age'].isnull() , 1, 0)

df_test['age_is_null'] = np.where(df_test['Age'].isnull() , 1, 0)
df_train.info()

df_test.info()
#Transforme les valeurs Age en type int (les valeurs Nan passe a 0)

df_train['Age'] = df_train['Age'].fillna(0.0).astype(int)

df_test['Age'] = df_test['Age'].fillna(0.0).astype(int)
df_train.info()

df_test.isna().sum()
#(df_train.loc[df_train['age_is_null'] == 1]).isna().sum()

#on récuperer les personnes qui n'ont pas d'age renseigné 

df_age_undefine = df_train.loc[df_train['age_is_null'] == 1]

df_test_age_undefine = df_test.loc[df_test['age_is_null'] == 1]



#On calcul l'age moyen avec les valeurs enseigné 

age_moyen = (df_train.loc[df_train['age_is_null'] == 0])['Age'].mean()

df_age_undefine['Age'] = age_moyen



test_age_moyen = (df_test.loc[df_test['age_is_null'] == 0])['Age'].mean()

df_test_age_undefine['Age'] = test_age_moyen
#Parse float into int columns Age

df_age_undefine['Age'] = df_age_undefine['Age'].fillna(0.0).astype(int)



df_test_age_undefine['Age'] = df_test_age_undefine['Age'].fillna(0.0).astype(int)
df_train.loc[df_train['age_is_null'] == 1] = df_age_undefine



df_test.loc[df_test['age_is_null'] == 1] = df_test_age_undefine
df_train['Embarked_undefine'] = np.where(df_train['Embarked'].isnull() , 1, 0)



df_test['Embarked_undefine'] = np.where(df_test['Embarked'].isnull() , 1, 0)
#Les plus riches survivent ou ceux qui ont acheté le billet le plus cher 

sns.catplot(x="Survived", y="Fare", data=df_train, kind='bar');
df_train
df_train['Fare_cat'] = pd.cut(df_train['Fare'], 5, labels=np.arange(5))

#df_train.loc[df_train['Fare_cat'].isna() == 1]



df_test['Fare_cat'] = pd.cut(df_test['Fare'], 5, labels=np.arange(5))



df_test_fare_undefine = df_test.loc[df_test['Fare_cat'].isna() == 1]

df_test_fare_undefine['Fare'] = df_test['Fare'].mean()



df_test.loc[df_test['Fare_cat'].isna() == 1] = df_test_fare_undefine



df_test['Fare_cat'] = pd.cut(df_test['Fare'], 5, labels=np.arange(5))

df_test.loc[df_test['Fare_cat'].isna() == 1]
NUM_FEATURES = ['Parch', 'SibSp']

CAT_FEATURES = ['Fare_cat', 'Sex']
# crée dataset avec les columns qui nous intéresse

X_train_NUM = df_train.loc[:, NUM_FEATURES]

X_train_cat = df_train.loc[:, CAT_FEATURES]
# on fusion le valeur numeraire et catégorielles

X_train = pd.concat([X_train_NUM, X_train_cat], axis=1)
# crée dataset avec les columns qui nous intéresse

X_test_NUM = df_test.loc[:, NUM_FEATURES]

X_test_cat = df_test.loc[:, CAT_FEATURES]
# on fusion le valeur numeraire et catégorielles

X_test = pd.concat([X_test_NUM, X_test_cat], axis=1)
y_train = df_train['Survived']
for c in X_train.select_dtypes('object').columns:

    X_train[c] = X_train[c].astype('category').cat.codes
for c in X_test.select_dtypes('object').columns:

    X_test[c] = X_test[c].astype('category').cat.codes
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score



rf = RandomForestClassifier(n_estimators=100)
X_test.isna().sum()
rf.fit(X_train, y_train)
#cellule pour stocker le résultat du modéle en appéllant son modéle rf et son set de test X_test

submission = pd.DataFrame({

    'PassengerId': df_test['PassengerId'],

    'Survived': rf.predict(X_test)

})



submission.to_csv('submission.csv', index=False)