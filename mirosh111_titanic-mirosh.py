#import all needed packages
import pandas as pd
import numpy as np
import csv as csv
import seaborn as sns

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.cross_validation import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

addpoly = True
plot_lc = 0 
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
# Смотрим как пол влияет на шанс спастись
print(df_train[['Sex', 'Survived']].groupby(['Sex']).mean())
sns.catplot(x='Sex', y='Survived',  kind='bar', data=df_train)
# Смотрим, как класс влияет на шанс спастись
print(df_train[['Pclass', 'Survived']].groupby(['Pclass'], 
                                                    as_index=False).mean().sort_values(by='Survived', ascending=False))
# Смотрим, как количесво родственников влияет на шанс спастись
sns.catplot(x='SibSp', y='Survived', data=df_train, kind='bar')
# Смотрим, как Parch влияет на шанс спастись 
print(df_train[["Parch", "Survived"]].groupby(['Parch'], 
                                                   as_index=False).mean().sort_values(by='Survived', ascending=False))
#Age
train_random_ages = np.random.randint(df_train["Age"].mean() - df_train["Age"].std(),
                                          df_train["Age"].mean() + df_train["Age"].std(),
                                          size = df_train["Age"].isnull().sum())

test_random_ages = np.random.randint(df_test["Age"].mean() - df_test["Age"].std(),
                                          df_test["Age"].mean() + df_test["Age"].std(),
                                          size = df_test["Age"].isnull().sum())

df_train["Age"][np.isnan(df_train["Age"])] = train_random_ages
df_test["Age"][np.isnan(df_test["Age"])] = test_random_ages
df_train['Age'] = df_train['Age'].astype(int)
df_test['Age']    = df_test['Age'].astype(int)

# Embarked 
df_train["Embarked"].fillna('S', inplace=True)
df_test["Embarked"].fillna('S', inplace=True)
df_train['Port'] = df_train['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
df_test['Port'] = df_test['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
del df_train['Embarked']
del df_test['Embarked']

# Fare
df_test["Fare"].fillna(df_test["Fare"].median(), inplace=True)

# Cabin меняем на Has_Cabin
df_train['Has_Cabin'] = df_train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
df_test['Has_Cabin'] = df_test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

all_data = [df_train,df_test]

# Так как выживаемость в одиночку выше, сделаем специальный признак IsAlone
for dataset in all_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

for dataset in all_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

for dataset in all_data:
    dataset['FamilySizeGroup'] = 'Small'
    dataset.loc[dataset['FamilySize'] == 1, 'FamilySizeGroup'] = 'Alone'
    dataset.loc[dataset['FamilySize'] >= 5, 'FamilySizeGroup'] = 'Big'
    
family_mapping = {"Small": 0, "Alone": 1, "Big": 2}

for dataset in all_data:
    dataset['FamilySizeGroup'] = dataset['FamilySizeGroup'].map(family_mapping)

    
# Меняем Age, Sex и Fare    
for dataset in all_data:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
        
for dataset in all_data:    
    dataset.loc[ dataset['Age'] <= 14, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 14) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4

for dataset in all_data:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

# Делаем еще один признак    
for dataset in all_data:
    dataset['IsChildandRich'] = 0
    dataset.loc[(dataset['Age'] <= 0) & (dataset['Pclass'] == 1 ),'IsChildandRich'] = 1  
    dataset.loc[(dataset['Age'] <= 0) & (dataset['Pclass'] == 2 ),'IsChildandRich'] = 1  

# Удаляем ненужные признаки

del df_train['Name']
del df_test['Name']

del df_train['SibSp']
del df_test['SibSp']

del df_train['Parch']
del df_test['Parch']

del df_train['FamilySize']
del df_test['FamilySize']

del df_train['Cabin']
del df_test['Cabin']

del df_train['Ticket']
del df_test['Ticket']

del df_train['Port']
del df_test['Port']
print('train dataset: %s, test dataset %s' %(str(df_train.shape), str(df_test.shape)) )
df_train.head()



del df_train['PassengerId']

X_train = df_train.drop("Survived",axis=1)
Y_train = df_train["Survived"]
X_test  = df_test.drop("PassengerId",axis=1).copy()

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)





# KNN
alg_KNN = KNeighborsClassifier(n_neighbors=3)
alg_KNN.fit(X_train,Y_train)

result_train = alg_KNN.score(X_train,Y_train)
result_val = cross_val_score(alg_KNN, X_train, Y_train, cv=5).mean()
print(result_train,result_val)
# Logistic Regression
logreg = LogisticRegression() 
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)

result_train = logreg.score(X_train, Y_train)
result_val = cross_val_score(logreg,X_train, Y_train, cv=5).mean()
print(result_train , result_val)
# RandomForest
random_forest = RandomForestClassifier(criterion='gini', 
                             n_estimators=1000,
                             min_samples_split=10,
                             min_samples_leaf=1,
                             max_features='auto',
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1)

seed= 42
random_forest =RandomForestClassifier(n_estimators=1000, criterion='entropy', max_depth=5, min_samples_split=2,
                           min_samples_leaf=1, max_features='auto',    bootstrap=False, oob_score=False, 
                           n_jobs=1, random_state=seed,verbose=0)

random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)

result_train = random_forest.score(X_train, Y_train)
result_val = cross_val_score(random_forest,X_train, Y_train, cv=5).mean()

print(result_train , result_val)
submission = pd.DataFrame({
        "PassengerId": df_test["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('titanic.csv', index=False)
