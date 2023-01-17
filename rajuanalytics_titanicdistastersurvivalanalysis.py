#This script considered with basics of Machine Learning..

#Model developed with GridSearchCV, entropy, jini, one-hot encoding, some feature engineering steps..

#I will keep modify this model to reach good percentage



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import tree

from sklearn import model_selection

from subprocess import check_output

import seaborn as sns

from sklearn import ensemble

from sklearn import preprocessing

titanic_train = pd.read_csv('../input/train.csv')

titanic_test = pd.read_csv('../input/test.csv')

titanic_train.info()

titanic_test['Survived'] = None

titanic_test.info()



titanicAll = pd.concat([titanic_train,titanic_test])

titanicAll.info()



#explore the dataframe

titanic_train.shape

titanic_train.info()

#EDA

titanicAll.shape

titanicAll.info



#create an instance of Imputer class with required arguments

mean_imputer = preprocessing.Imputer()

#compute mean of age and fare respectively

mean_imputer.fit(titanic_train[['Age','Fare']])

#fill up the missing data with the computed means 

titanicAll[['Age','Fare']] = mean_imputer.transform(titanicAll[['Age','Fare']])



#Feature Considered Till now Age, Fare, Survived

#Feature Creation: Creating new feature with Age column to see visualization. To find differences in age groups.

def ageRange(age):

    ageRange=''

    if age<=10:

        ageRange='Child'

    elif age<=30:

        ageRange='Young'

    elif(age<=50):

        ageRange='Adult'

    elif(age<=80):

        ageRange='Old'

    elif(age<=100):

        ageRange='Oldest'

    return ageRange



titanicAll['Age1'] = titanicAll['Age'].map(ageRange)

titanicAll.groupby('Age1').size()

#titanic_train.describe(['Pclass'])

#Now visualize age groups/ranges

sns.FacetGrid(data=titanicAll,row='Survived', size=8).map(sns.countplot, 'Age1').add_legend()

titanicAll.groupby(['Survived','Age1']).size()

sns.factorplot(x="Survived", hue="Age1", data=titanicAll, kind="count", size=6)

#In grapch there is much deviation in Age1 created, so I will use Title for Model Building

#Features Considered Till now (+Age1, Fare, Survived)(-Age)

#Feature Creation: Creating new feature by using Name column



def titleExtractionByName(name):

    return (name.split(',')[1]).split('.')[0].strip()

    

titanicAll['Title'] = titanicAll['Name'].map(titleExtractionByName)



titanicAll.groupby('Title').size()



titleDict = {'Capt': 'Officer',

             'Col': 'Officer',

             'Dr':'Officer',

             'Major':'Officer',

             'Rev':'Officer',

             'Sir':'Royalty',

             'the Countess':'Royalty',

             'Don': 'Royalty',

             'Jonkheer':'Royalty',

             'Lady':'Royalty',

             'Master':'Master',

             'Miss':'Miss',

             'Mlle':'Miss',

             'Mrs':'Mrs',

             'Mme':'Mrs',

             'Ms':'Mrs',

             'Mr':'Mr'

             }



titanicAll['Title'] = titanicAll['Title'].map(titleDict)

titanicAll.groupby('Title').size()

#Now visualize Title groups/ranges



sns.FacetGrid(data=titanicAll,row='Survived', size=8).map(sns.countplot, 'Title').add_legend()

#With two columns FacetGrid will not help much. Better go for BiVariate-Categorical

titanicAll.groupby(['Survived','Sex']).size()

sns.factorplot(x="Survived", hue="Title", data=titanicAll, kind="count", size=6)

#In grapch there is much deviation in Title created, so I will use Title for Model Building

#Features Considered Till now (+Age1,+Title, Fare, Survived)(-Age,-Name)

#Data Preparation

#fill the missing value for fare column

titanic_test.loc[titanic_test['Fare'].isnull() == True, 'Fare'] = titanic_test['Fare'].mean()

titanic_test1 = pd.get_dummies(titanic_test, columns=['Sex','Pclass','Embarked'])

titanic_test1.shape

titanic_test1.info()

#Exploration of SibSp(Sibling Spuse) and Parch(Parent Children) - Here we need domain knowledge.

titanicAll['SibSp'].describe()

titanicAll['Parch'].describe()

#Subling, Spouse, Parent Children are related to family, for a given family how many are travelling.

#Lets create a Feature 'Family', add(SibSp, Parch)

titanicAll['Family'] = titanicAll['SibSp']+titanicAll['Parch']

titanicAll['Family'].describe()

#Now Family Feature is seems continuous

#Lets make categorical



def familySize(family):

    familySize = ''

    if(family<=1):

        familySize = 'Single'

    elif(family<=3):

        familySize = 'Small'

    elif(family<=5):

        familySize = 'Medium'

    else:

        familySize = 'Large'

    return familySize



titanicAll['FamilySize'] = titanicAll['Family'].map(familySize)

#Now Lets visualize 

titanicAll.groupby(['FamilySize','Survived']).size()

sns.factorplot(x="Survived", hue="FamilySize", data=titanicAll, kind="count", size=6)

#From Graph there is much deviation between family sizes, so Feature 'FamilySize' is important feature, So I am considering for Model building.

#Features Considered Till now (+Age1,+Title,+FamilySize, Fare, Survived)(-Age,-Name,-SibSp,-Parch,-Family)



titanicAll.describe()



#convert categorical columns to one-hot encoded columns

titanic1 = pd.get_dummies(titanicAll, columns=['Sex','Pclass','Embarked', 'Age1', 'Title', 'FamilySize'])

titanic1.shape

titanic1.info()

#One Hot Encoding

#dropping us used Feautres from train data

titanic2 = titanic1.drop(['PassengerId','Name','Age','Ticket','Cabin','Survived','SibSp','Parch'], axis=1, inplace=False)

titanic2.info()

X_train = titanic2[0:titanic_train.shape[0]]

X_train.shape

X_train.info()

y_train = titanic_train['Survived']

#Model Building

#oob scrore is computed as part of model construction process

rf_estimator = ensemble.RandomForestClassifier(oob_score=True, random_state=2017)

rf_grid = {'n_estimators':[50,100], 'max_features':[4,5,6], 'max_depth':[3,4,5,6,7]}

grid_rf_estimator = model_selection.GridSearchCV(rf_estimator, rf_grid, cv=10,n_jobs=1)

grid_rf_estimator.fit(X_train, y_train)

print(grid_rf_estimator.grid_scores_)

print(grid_rf_estimator.best_score_)#83

print(grid_rf_estimator.best_params_)

grid_rf_estimator.best_estimator_.oob_score_#83

print(grid_rf_estimator.score(X_train, y_train))#83

##################Final Prections Preparation

X_test = titanic2[titanic_train.shape[0]:]

X_test.shape

X_test.info()



titanic_test['Survived'] = grid_rf_estimator.predict(X_test)



#titanic_test.to_csv('submission.csv', columns=['PassengerId','Survived'],index=False)

#@Kaggle, got 79.425, I will keep update the model. Please give valuable comments. 

#I will consider your comments in next version.

#Thanks
