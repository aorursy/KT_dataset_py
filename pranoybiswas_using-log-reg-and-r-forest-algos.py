# pandas

import pandas as pd

from pandas import Series,DataFrame



# numpy, matplotlib, seaborn

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier
# get titanic & test csv files as a DataFrame

titanic_df = pd.read_csv("../input/train.csv")

test_df    = pd.read_csv("../input/test.csv")



# preview the data

titanic_df



titanic_df = titanic_df.drop(['PassengerId','Name','Ticket','Embarked','Cabin'], axis=1)

test_df    = test_df.drop(['Name','Ticket','Embarked','Cabin'], axis=1)



titanic_df.head()





# Fare



# only for test_df, since there is a missing "Fare" values

test_df["Fare"].fillna(test_df["Fare"].median(), inplace=True)



# convert from float to int

titanic_df['Fare'] = titanic_df['Fare'].astype(int)

test_df['Fare']    = test_df['Fare'].astype(int)



# Family



# Instead of having two columns Parch & SibSp, 

# we can have only one column represent if the passenger had any family member aboard or not,

# Meaning, if having any family member(whether parent, brother, ...etc) will increase chances of Survival or not.

titanic_df['Family'] =  titanic_df["Parch"] + titanic_df["SibSp"]

titanic_df['Family'].loc[titanic_df['Family'] > 0] = 1

titanic_df['Family'].loc[titanic_df['Family'] == 0] = 0



test_df['Family'] =  test_df["Parch"] + test_df["SibSp"]

test_df['Family'].loc[test_df['Family'] > 0] = 1

test_df['Family'].loc[test_df['Family'] == 0] = 0



# drop Parch & SibSp

titanic_df = titanic_df.drop(['SibSp','Parch'], axis=1)

test_df    = test_df.drop(['SibSp','Parch'], axis=1)





# Sex



# As we see, children(age < ~16) on aboard seem to have a high chances for Survival.

# So, we can classify passengers as males, females, and child

def get_person(passenger):

    age,sex = passenger

    return 'child' if age < 16 else sex

    

titanic_df['Person'] = titanic_df[['Age','Sex']].apply(get_person,axis=1)

test_df['Person']    = test_df[['Age','Sex']].apply(get_person,axis=1)



# No need to use Sex column since we created Person column

titanic_df.drop(['Sex'],axis=1,inplace=True)

test_df.drop(['Sex'],axis=1,inplace=True)



# create dummy variables for Person column

person_dummies_titanic  = pd.get_dummies(titanic_df['Person'])

person_dummies_titanic.columns = ['Child','Female','Male']

person_dummies_test  = pd.get_dummies(test_df['Person'])

person_dummies_test.columns = ['Child','Female','Male']

titanic_df = titanic_df.join(person_dummies_titanic)

test_df    = test_df.join(person_dummies_test)



titanic_df.drop(['Person'],axis=1,inplace=True)

test_df.drop(['Person'],axis=1,inplace=True)



# Pclass



# create dummy variables for Pclass column

pclass_dummies_titanic  = pd.get_dummies(titanic_df['Pclass'])

pclass_dummies_titanic.columns = ['Class_1','Class_2','Class_3']

pclass_dummies_test  = pd.get_dummies(test_df['Pclass'])

pclass_dummies_test.columns = ['Class_1','Class_2','Class_3']



titanic_df.drop(['Pclass'],axis=1,inplace=True)

test_df.drop(['Pclass'],axis=1,inplace=True)



titanic_df = titanic_df.join(pclass_dummies_titanic)

test_df    = test_df.join(pclass_dummies_test)



titanic_df.head()
from sklearn.ensemble import RandomForestRegressor

#predicting missing values in age using Random Forest

def fill_missing_age(df):

    

    #Feature set

    age_df = df[['Age','Fare', 'Family','Child','Female','Male','Class_1','Class_2','Class_3']]

    # Split sets into train and test

    train  = age_df.loc[ (df.Age.notnull()) ]# known Age values

    test = age_df.loc[ (df.Age.isnull()) ]# null Ages

    

    # All age values are stored in a target array

    y = train.values[:, 0]

    

    # All the other values are stored in the feature array

    X = train.values[:, 1::]

    

    # Create and fit a model

    rtr = RandomForestRegressor(n_estimators=2000, n_jobs=-1)

    rtr.fit(X, y)

    

    # Use the fitted model to predict the missing values

    predictedAges = rtr.predict(test.values[:, 1::])

    

    # Assign those predictions to the full data set

    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges 

    

    return df



titanic_df=fill_missing_age(titanic_df)

test_df=fill_missing_age(test_df)



titanic_df

from sklearn import preprocessing



std_scale = preprocessing.StandardScaler().fit(titanic_df[['Age', 'Fare']])

titanic_df[['Age', 'Fare']] = std_scale.transform(titanic_df[['Age', 'Fare']])





std_scale = preprocessing.StandardScaler().fit(test_df[['Age', 'Fare']])

test_df[['Age', 'Fare']] = std_scale.transform(test_df[['Age', 'Fare']])



titanic_df.head()
# define training and testing sets



X_train = titanic_df.drop("Survived",axis=1)

Y_train = titanic_df["Survived"]

X_test  = test_df.drop("PassengerId",axis=1).copy()
# Logistic Regression

from sklearn import cross_validation

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import ShuffleSplit



logreg = LogisticRegression()



logreg.fit(X_train, Y_train)



Y_pred = logreg.predict(X_test)



# Compute the accuracy score for all the cross validation folds.

cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=50)



scores = cross_val_score(logreg, X_train, Y_train,scoring='f1', cv=cv)



scores.mean()
# Random Forests



random_forest = RandomForestClassifier(n_estimators=100)



random_forest.fit(X_train, Y_train)



Y_pred = random_forest.predict(X_test)



# Compute the accuracy score for all the cross validation folds.

cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=50)



scores = cross_val_score(random_forest, X_train, Y_train,scoring='f1', cv=cv)



scores.mean()
submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('titanic.csv', index=False)