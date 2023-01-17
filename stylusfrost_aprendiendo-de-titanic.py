import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from pandas import Series,DataFrame

# machine learning

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegressionCV

from sklearn.cross_validation import KFold

from sklearn.grid_search import GridSearchCV

from sklearn.metrics import accuracy_score, f1_score, precision_score

from sklearn.metrics import recall_score, classification_report, confusion_matrix


#it has null?

def pHasNullValues(df):

    print('hasNull\tCount\t\tFeature');

    for column in list(df.columns.values):

        if df[column].isnull().values.any()>0:

            print('Yes\t',df[column].isnull().sum(),'\t\t',column)

        else:

            print('No\t\t\t',column)

            

    return 



def isChildWithAge(age,sex):

    if float(age)<14.0:

        return 'Child'

    else:

        return sex

    





#Cabin and Class to level

def cabin2level(cabin):

    if cabin==np.nan:

        return 0

    

    

    if cabin.find('A')!=-1:

        return 1

    if cabin.find('B')!=-1:

        return 2

    if cabin.find('C')!=-1:

        return 3

    if cabin.find('D')!=-1:

        return 4

    if cabin.find('E')!=-1:

        return 5

    if cabin.find('F')!=-1:

        return 6

    if cabin.find('G')!=-1:

        return 7

    return 0

    
#Carga de datos

train_df = pd.read_csv("../input/train.csv", dtype={"Age": np.float64} )

test_df  = pd.read_csv("../input/test.csv", dtype={"Age": np.float64} )



#Juntamos los datos

total_df = pd.concat([train_df,test_df])


train_df['Ticket']
train_df.head()
train_df.info()

print("----------------------------")

test_df.info()

print("----------------------------")

total_df.info()
#Tenemos alguna variable a Nula?

print('Train data: Total',len(train_df));

print();

pHasNullValues(train_df);

print();

print('Test data:  Total',len(test_df));

print();

pHasNullValues(test_df);

print();

print('Total data:  Total',len(total_df));

print();

pHasNullValues(total_df);
#Vamos a quitar lo que no interesa

train_df.drop(["PassengerId"],axis=1,inplace=True)
#Vamos a solucionar los campos nulos



# Age

# Aplicamos la funcion que decidamos

# Mediam

train_df["Age"].fillna(total_df["Age"].median(), inplace=True)

test_df["Age"].fillna(total_df["Age"].median(), inplace=True)



# Fare

# Aplicamos la funcion que decidamos

# Mediam

train_df['Fare'].fillna(total_df['Fare'].median(), inplace=True)

test_df['Fare'].fillna(total_df['Fare'].median(), inplace=True)



# Cabin

# Aplicamos la funcion que decidamos

# Vamos a intentar convertirlo en el número de planta 

# Parece que el primer dígito segun https://www.encyclopedia-titanica.org/titanic-deckplans

# A es la planta más cercana a la cubierta  y G la más lejana



train_df['Cabin']=train_df['Cabin'].astype(str).apply(cabin2level).astype(int)

test_df['Cabin']=test_df['Cabin'].astype(str).apply(cabin2level).astype(int)





# Embarked

# Aplicamos la funcion que decidamos

# Asignamos la salida desde la cual ha embarcado más gente

pd.value_counts(train_df['Embarked'].values, sort=True)

# S Es el valor 

train_df['Embarked'].fillna('S', inplace=True)


train_df['Sex']=train_df[['Age','Sex']].apply(lambda row: isChildWithAge(row['Age'], row['Sex']), axis=1)

test_df['Sex']=test_df[['Age','Sex']].apply(lambda row: isChildWithAge(row['Age'], row['Sex']), axis=1)
# Si esta todo bien ya no tenemos nulos

print('Train data: Total',len(train_df));

print();

pHasNullValues(train_df);

print();

print('Test data:  Total',len(test_df));

print();

pHasNullValues(test_df);
#¿Que hacemos con las features que no son numéricas?

#Sex











# create dummy variables for Sex

person_dummies_train  = pd.get_dummies(train_df['Sex'])

person_dummies_train.columns = ['Female','Male','Child']

person_dummies_train.drop(['Male'], axis=1, inplace=True)



person_dummies_test  = pd.get_dummies(train_df['Sex'])

person_dummies_test.columns = ['Female','Male','Child']

person_dummies_test.drop(['Male'], axis=1, inplace=True)



train_df = train_df.join(person_dummies_train)

test_df  = test_df.join(person_dummies_test)



train_df.drop(['Sex'],axis=1,inplace=True)

test_df.drop(['Sex'],axis=1,inplace=True)



# Embarked

# create dummy variables Embarked 

embark_dummies_train  = pd.get_dummies(train_df['Embarked'])

embark_dummies_train.drop(['S'], axis=1, inplace=True)



embark_dummies_test  = pd.get_dummies(test_df['Embarked'])

embark_dummies_test.drop(['S'], axis=1, inplace=True)



train_df = train_df.join(embark_dummies_train)

test_df    = test_df.join(embark_dummies_test)



train_df.drop(['Embarked'], axis=1,inplace=True)

test_df.drop(['Embarked'], axis=1,inplace=True)



# Name

train_df.drop(['Name'], axis=1,inplace=True)

test_df.drop(['Name'], axis=1,inplace=True)



# Ticket

train_df.drop(['Ticket'], axis=1,inplace=True)

test_df.drop(['Ticket'], axis=1,inplace=True)

train_df.head()


train_df['Age'] = (train_df['Age'] - train_df['Age'].mean()) / (train_df['Age'].max() - train_df['Age'].min())

train_df['Fare'] = (train_df['Fare'] - train_df['Fare'].mean()) / (train_df['Fare'].max() - train_df['Fare'].min())

train_df['Pclass'] = (train_df['Pclass'] - train_df['Pclass'].mean()) / (train_df['Pclass'].max() - train_df['Pclass'].min())

train_df['SibSp'] = (train_df['SibSp'] - train_df['SibSp'].mean()) / (train_df['SibSp'].max() - train_df['SibSp'].min())

train_df['Female'] = (train_df['Female'] - train_df['Female'].mean()) / (train_df['Female'].max() - train_df['Female'].min())

train_df['Parch'] = (train_df['Parch'] - train_df['Parch'].mean()) / (train_df['Parch'].max() - train_df['Parch'].min())

train_df['C'] = (train_df['C'] - train_df['C'].mean()) / (train_df['C'].max() - train_df['C'].min())

train_df['Q'] = (train_df['Q'] - train_df['Q'].mean()) / (train_df['Q'].max() - train_df['Q'].min())

train_df['Cabin'] = (train_df['Cabin'] - train_df['Cabin'].mean()) / (train_df['Cabin'].max() - train_df['Cabin'].min())

train_df['Child'] = (train_df['Child'] - train_df['Child'].mean()) / (train_df['Child'].max() - train_df['Child'].min())
train_df.head()
# define training, validation and testing sets





X_train = train_df.drop(["Survived"],axis=1)

Y_train = train_df["Survived"]

X_test  = test_df.drop("PassengerId",axis=1).copy()
# ¿Lo podemos mejorar?


# Random Forests



n_range = range(10, 100, 10)

param_grid = dict(n_estimators=list(n_range),criterion = ["gini", "entropy"])

rfc = RandomForestClassifier(n_estimators=20)

grid = GridSearchCV(rfc, param_grid, cv=10, scoring='accuracy')

grid.fit(X_train, Y_train)

print(grid.best_score_ , grid.best_params_)



Y_pred = grid.predict(X_test)



#random_forest = RandomForestClassifier(n_estimators=100)



#random_forest.fit(X_train, Y_train)



#Y_pred = random_forest.predict(X_test)



#random_forest.score(X_train, Y_train)
# Train Validation Test
submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('titanic.csv', index=False)