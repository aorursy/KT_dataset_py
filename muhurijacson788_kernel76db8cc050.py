import pandas as pd

import numpy as np



import seaborn as sns

import matplotlib.pyplot as plt



# ignore warnings

import warnings

warnings.simplefilter(action='ignore')
titanic = pd.read_csv('../input/titanic_train.csv')

test = pd.read_csv('../input/titanic_test.csv')

print(titanic.shape)

titanic.head(2)
#check columns with null values

titanic.isnull().sum()
#Replace null values in age with the mean of all ages

titanic['Age']=titanic['Age'].fillna(titanic['Age'].mean())
#Replace null values in Embarked with the most frequent value of Embarked

titanic['Embarked']=titanic['Embarked'].astype(str).fillna(titanic['Embarked'].mode())
#a huge percentage of Cabin (above 60%) has null values, drop cabin together with other columns not needed in training



titanic.drop(['Cabin','PassengerId','Ticket'],axis=1,inplace=True)
#Confirm all null values have been fixed

titanic.isnull().sum()
titanic.head(2)
sns.barplot(x=titanic.Pclass,y=titanic.Survived)
titanic['PassengerTitle']=titanic['Name'].str.split(',',expand=True)[1].str.split('.', expand=True)[0]

titanic.drop('Name',axis=1,inplace=True)

titanic['PassengerTitle'].value_counts()
sns.barplot(x=titanic['PassengerTitle'],y=titanic['Survived'])
sns.barplot(x=titanic.Sex,y=titanic.Survived)
sns.boxplot(x=titanic['Age'])
age_categories=[]

for age in titanic['Age']:

    if age<18:

        age_category='Minor'

    elif age>18 and age<40:

        age_category='Youth'

    elif age>40 and age<60:

        age_category='Middle Age'

    else:

        age_category='Senior'

    age_categories.append(age_category)

titanic['AgeCategory']=age_categories
sns.barplot(x=titanic.AgeCategory,y=titanic.Survived)
titanic['FamilyTotal']=titanic['SibSp']+titanic['Parch']+1

titanic['If_Alone']=titanic['FamilyTotal']<2
sns.barplot(x=titanic.If_Alone,y=titanic.Survived)
#Summary statistics for fares

titanic['Fare'].describe()
#Find fares outliers

sns.boxplot(x=titanic['Fare'])
titanic = titanic[titanic['Fare'] < 250]
#Save the target column before dropping it from the training set

target=titanic['Survived'].to_list()
#drop the target column from the trainig set

titanic.drop('Survived',axis=1,inplace=True)
titanic.reset_index(drop=True)

print(len(target))

print(titanic.shape)

titanic.head()
from sklearn import preprocessing



encode=preprocessing.LabelEncoder()

titanic['Sex']=encode.fit_transform(titanic['Sex'])

titanic['Embarked']=encode.fit_transform(titanic['Embarked'])

titanic['PassengerTitle']=encode.fit_transform(titanic['PassengerTitle'])

titanic['AgeCategory']=encode.fit_transform(titanic['AgeCategory'])

titanic['If_Alone']=encode.fit_transform(titanic['If_Alone'])
titanic.head()
from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test=train_test_split(titanic,target,test_size=0.2)#Split data using a test size of 20%

print(X_train.shape)

print(len(y_train))
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score





clf_logreg=LogisticRegression()

clf_logreg.fit(X_train,y_train)



logreg_predictions=clf_logreg.predict(X_test)



logreg_score=accuracy_score(y_test,logreg_predictions)

logreg_score
from sklearn.neighbors import KNeighborsClassifier



clf_knn=KNeighborsClassifier(n_neighbors=10)

clf_knn.fit(X_train,y_train)



knn_predictions=clf_knn.predict(X_test)



knn_score=accuracy_score(y_test,knn_predictions)

knn_score
from sklearn.tree import DecisionTreeClassifier



clf_dec=DecisionTreeClassifier()

clf_dec.fit(X_train,y_train)



dec_predictions=clf_dec.predict(X_test)



dec_score=accuracy_score(y_test,dec_predictions)

dec_score
from sklearn.model_selection import cross_val_score



cross_val_score(clf_logreg,X_train,y_train,cv=10,scoring='accuracy').mean()
cross_val_score(clf_knn,X_train,y_train,cv=10,scoring='accuracy').mean()
cross_val_score(clf_dec,X_train,y_train,cv=5,scoring='accuracy').mean()
from sklearn.model_selection import GridSearchCV

from numpy import random



# penalty=['l1', 'l2', 'elasticnet', 'none']

# dual=[True,False]

# tol=list(random.rand(0,2))

C=[1.0,1.5,2.0,2.5,3.0]

fit_intercept=[True,False]

# fit_intercept=[1.0,1.5,2.0,2.5,3.0]

class_weight=['balanced',None]

solver=['lbfgs', 'liblinear', 'sag', 'saga']

# multi_class=['auto', 'ovr', 'multinomial']

max_iter=[10,20,30,40,50,60,70,80,90,100,110,120,130,140,150]



param_grid=dict(C=C,fit_intercept=fit_intercept,\

                class_weight=class_weight,

                solver=solver,max_iter=max_iter)

print(param_grid)
grid=GridSearchCV(clf_logreg, param_grid, cv=10, scoring='accuracy', return_train_score=False)

grid.fit(X_train,y_train)
# view the results

pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score', 'params']]
# examine the best model

print(grid.best_score_)

print(grid.best_params_)
print(test.shape)

test.head(2)
test.isnull().sum()
test['Age']=test['Age'].fillna(test['Age'].mean())
age_categories=[]

for age in test['Age']:

    if age<18:

        age_category='Minor'

    elif age>18 and age<40:

        age_category='Youth'

    elif age>40 and age<60:

        age_category='Middle Age'

    else:

        age_category='Senior'

    age_categories.append(age_category)

test['AgeCategory']=age_categories
test['Fare']=test['Fare'].fillna(test['Fare'].mean())
test['PassengerTitle']=test['Name'].str.split(',',expand=True)[1].str.split('.', expand=True)[0]

test.drop('Name',axis=1,inplace=True)
test['FamilyTotal']=test['SibSp']+test['Parch']+1

test['If_Alone']=test['FamilyTotal']<2
#Set the passengerid column before dropping it from the test set

ids=test['PassengerId'].to_list()
test.drop(['Cabin','Ticket','PassengerId'],axis=1,inplace=True)
test['Sex']=encode.fit_transform(test['Sex'])

test['Embarked']=encode.fit_transform(test['Embarked'])

test['PassengerTitle']=encode.fit_transform(test['PassengerTitle'])

test['AgeCategory']=encode.fit_transform(test['AgeCategory'])

test['If_Alone']=encode.fit_transform(test['If_Alone'])
test.head(3)
print(grid.best_score_)

print(grid.best_params_)
#Insert the best parameter as shown by GridSearchCv



clf_logreg=LogisticRegression(C=3.0,class_weight=None,fit_intercept=True,max_iter=140,solver='lbfgs')



#Fit the model again but this time use the entire titanic dataset without splitting

clf_logreg.fit(titanic,target)
results=clf_logreg.predict(test)
#Create a dataframe of the outcome

submission=pd.DataFrame({'PassengerId':ids,'Survived':results})

submission.head(10)
#View counts of outcomes

submission['Survived'].value_counts()
#Create a csv file of the outcome in the current working directory

submission.to_csv('kaggle_titanic_results.csv',index=False)