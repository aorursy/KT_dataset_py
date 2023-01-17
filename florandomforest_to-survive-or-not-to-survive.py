# Imports

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
# Load datasets

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')



# ¨Preview

train_df.head(3)
train_df.info()

test_df.info()
# Suppression des colonnes inutiles

train_df.drop(['PassengerId',],axis=1,inplace=True)
train_df.drop(['Cabin','Embarked','Ticket'],axis=1,inplace=True)

test_df.drop(['Cabin','Embarked','Ticket'],axis=1,inplace=True)
# Création de deux facteurs 'girl' et 'boy' pour une colonne "Genre"

def get_genre(passenger):

    age,sex = passenger

    if age < 18 and sex == 'female':

        return 'girl'

    elif age < 18:

        return 'boy'

    else:

        return sex

    

train_df['Genre'] = train_df[['Age','Sex']].apply(get_genre,axis=1)

test_df['Genre']  = test_df[['Age','Sex']].apply(get_genre,axis=1)

# Plus besoin de colonne Sex

train_df.drop(['Sex'],axis=1,inplace=True)

test_df.drop(['Sex'],axis=1,inplace=True)



# Conversion colonne genre en variables dummy

sex_dummies_train  = pd.get_dummies(train_df['Genre'])

sex_dummies_train.drop(['male'], axis=1, inplace=True)

sex_dummies_test  = pd.get_dummies(test_df['Genre'])

sex_dummies_test.drop(['male'], axis=1, inplace=True)



# Ajout des dummies

train_df = train_df.join(sex_dummies_train)

test_df  = test_df.join(sex_dummies_test)



# Plus besoin de colonne Sex

train_df.drop(['Genre'],axis=1,inplace=True)

test_df.drop(['Genre'],axis=1,inplace=True)
# Family size

train_df['FamilySize'] = train_df['SibSp']+train_df['Parch']

test_df['FamilySize'] = test_df['SibSp']+test_df['Parch']
train_df['NameLength'] = train_df['Name'].apply(lambda x: len(x))

test_df['NameLength'] = test_df['Name'].apply(lambda x: len(x))



# Plus besoin de colonne Name

train_df.drop(['Name'],axis=1,inplace=True)

test_df.drop(['Name'],axis=1,inplace=True)
train_df.head(3)
# Train

mean_age_train = train_df['Age'].mean()

std_age_train  = train_df['Age'].std()

print('Mean age (train) : {}'.format(mean_age_train))

print('Std age (train) : {}'.format(std_age_train))

n_na_train = train_df['Age'].isnull().sum()

train_rand_age = np.random.randint(mean_age_train-std_age_train,mean_age_train+std_age_train,size=n_na_train)



train_df["Age"][np.isnan(train_df["Age"])] = train_rand_age



# Test

mean_age_test = test_df['Age'].mean()

std_age_test  = test_df['Age'].std()

#print('Mean age (test) : {}'.format(mean_age_test))

#print('Std age (test) : {}'.format(std_age_test))

n_na_test = test_df['Age'].isnull().sum()

test_rand_age = np.random.randint(mean_age_test-std_age_test,mean_age_test+std_age_test,size=n_na_test)



test_df["Age"][np.isnan(test_df["Age"])] = test_rand_age
med_fare_test = test_df['Fare'].median()

print('Median fare (test) : {}'.format(med_fare_test))

test_df['Fare'] = test_df['Fare'].fillna(med_fare_test)
### Définition des train/test set

X_train = train_df.drop(['Survived'],axis=1)

Y_train = train_df['Survived']



X_test = test_df.drop(['PassengerId'],axis=1)
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier



RF = RandomForestClassifier()



param_grid = {'n_estimators': [100, 200, 300, 400], 'max_features': [2, 3, 4, 5]}

model = GridSearchCV(

    estimator=RF, param_grid=param_grid, verbose=10

)

model.fit(X_train, Y_train)



model.score(X_train, Y_train)
model.best_estimator_
from sklearn.svm import SVC

#SVM_lin = SVC(kernel='linear')

#SVM_lin.fit(X_train,Y_train)

SVM_rbf = SVC(kernel='rbf')

SVM_rbf.fit(X_train,Y_train)



#print(SVM_lin.score(X_train,Y_train))

print(SVM_rbf.score(X_train,Y_train))
Y_pred = model.predict(X_test)
submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('titanic.csv', index=False)