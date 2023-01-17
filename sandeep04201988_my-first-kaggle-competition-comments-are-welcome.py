#Author:Sandeep Ramesh



# Importing the libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

from sklearn.preprocessing import Imputer

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC



# Importing the dataset

titanic_train = pd.read_csv('../input/train.csv')

titanic_test= pd.read_csv('../input/test.csv')

titanic_train.info()

titanic_test.info()

titanic_train.describe()
#plotting the scatter matrix first

import matplotlib.pyplot as plt

from pandas.tools.plotting import scatter_matrix

scatter_matrix(titanic_train, figsize=(25,25))

plt.show()
#dropping the columns which might not affect prediction

titanic_train=titanic_train.drop(['PassengerId','Ticket'],1)

titanic_test=titanic_test.drop(['Ticket'],1)



#To convert Sex to category datatype

titanic_train['Sex']=titanic_train['Sex'].astype('category')

titanic_test['Sex']=titanic_test['Sex'].astype('category')



#drop cabin because of too many NaN values

titanic_train= titanic_train.drop(['Cabin'],1)

titanic_test= titanic_test.drop(['Cabin'],1)



#heatmap with correlation 

sns.heatmap(titanic_train.corr())
#filling the Embarked with most frequent column after getting a count

#sub=pd.DataFrame(titanic_train)

#sub.to_csv('newtrain.csv')

sns.factorplot('Embarked',kind='count',data=titanic_train)

titanic_train['Embarked'] = titanic_train['Embarked'].fillna("S")

titanic_test['Embarked'] = titanic_test['Embarked'].fillna("S")



#converting Embarked to dummy variable and dropping the extra column 'S' for dummy trap since S has low chance

sns.factorplot('Embarked','Survived',data=titanic_train)

titanic_train=titanic_train.join(pd.get_dummies(titanic_train.Embarked,prefix='Embarked'))

titanic_train=titanic_train.drop(['Embarked','Embarked_S'],1)

titanic_test=titanic_test.join(pd.get_dummies(titanic_test.Embarked,prefix='Embarked'))

titanic_test=titanic_test.drop(['Embarked','Embarked_S'],1)
#missing values in Age and replace with median

imputer = Imputer(missing_values='NaN',strategy='median',axis=0)

imputer = imputer.fit(titanic_train[['Age']])

titanic_train[['Age']]=imputer.transform(titanic_train[['Age']])

imputer = imputer.fit(titanic_test[['Age']])

titanic_test[['Age']]=imputer.transform(titanic_test[['Age']])







#missing values in fare and replace with median

imputer = Imputer(missing_values='NaN',strategy='median',axis=0)

imputer = imputer.fit(titanic_train[['Fare']])

titanic_train[['Fare']]=imputer.transform(titanic_train[['Fare']])

imputer = imputer.fit(titanic_test[['Fare']])

titanic_test[['Fare']]=imputer.transform(titanic_test[['Fare']])





#To convert Sex from categorical to dummy variables and determine which gender to drop for dummy variable trap

sns.factorplot('Sex','Survived',data=titanic_train)

titanic_train=titanic_train.join(pd.get_dummies(titanic_train.Sex,prefix='Sex'))

titanic_train=titanic_train.drop(['Sex','Sex_male'],1)

titanic_test=titanic_test.join(pd.get_dummies(titanic_test.Sex,prefix='Sex'))

titanic_test=titanic_test.drop(['Sex','Sex_male'],1)
#converting Pclass categorical to dummy variables and determine which class to drop for dummy variable trap

#titanic_train['Pclass']=pd.get_dummies(titanic_train.Pclass)

sns.factorplot('Pclass','Survived',data=titanic_train)

plt.hist(titanic_train.Pclass)    #to determine class 2 or class 3 to drop

titanic_train=titanic_train.join(pd.get_dummies(titanic_train.Pclass,prefix='Pclass'))

titanic_train=titanic_train.drop(['Pclass','Pclass_3'],1)

titanic_test=titanic_test.join(pd.get_dummies(titanic_test.Pclass,prefix='Pclass'))

titanic_test=titanic_test.drop(['Pclass','Pclass_3'],1)
#creating a new feature Title and replace least used titles to commonly used titles

titanic_train['Title']=titanic_train.Name.map(lambda x: x.split(',')[1].split('.')[0].strip())

unique=titanic_train.Title.unique().astype('str')

titanic_train['Title']=titanic_train.Title.replace(to_replace=['Don','Rev','Dr','Major','Sir','Col','Capt','Jonkheer'],value='Mr')

titanic_train['Title']=titanic_train.Title.replace(to_replace=['the Countess','Lady','Mlle'],value='Mrs')

titanic_train['Title']=titanic_train.Title.replace(to_replace=['Ms','Mme'],value='Miss')

titanic_train.Title.value_counts()



titanic_test['Title']=titanic_test.Name.map(lambda x: x.split(',')[1].split('.')[0].strip())

unique1=titanic_test.Title.unique().astype('str')

titanic_test['Title']=titanic_test.Title.replace(to_replace=['Don','Rev','Dr','Major','Sir','Col','Capt','Jonkheer'],value='Mr')

titanic_test['Title']=titanic_test.Title.replace(to_replace=['the Countess','Lady','Mlle','Dona'],value='Mrs')

titanic_test['Title']=titanic_test.Title.replace(to_replace=['Ms','Mme'],value='Miss')

titanic_test.Title.value_counts()



#drop the name feature column

titanic_train=titanic_train.drop(['Name'],axis=1)

titanic_test=titanic_test.drop(['Name'],axis=1)



#dropping the title 'Mr' to avoid dummy variable trap

sns.factorplot(x="Title",y="Survived",data=titanic_train)

titanic_train=titanic_train.join(pd.get_dummies(titanic_train.Title,prefix='Title'))

titanic_train=titanic_train.drop(['Title','Title_Mr'],1)

titanic_test=titanic_test.join(pd.get_dummies(titanic_test.Title,prefix='Title'))

titanic_test=titanic_test.drop(['Title','Title_Mr'],1)
#Preparing the dataset for train and test

x_train=titanic_train.drop(['Survived'],1)

y_train=titanic_train['Survived']

x_test=titanic_test.drop(['PassengerId'],1)





#Fitting and predicting the various Machine Learning Algorithms



lr=LogisticRegression(random_state=0)

lr=lr.fit(x_train,y_train)

y_pred_lr=lr.predict(x_test)

score=lr.score(x_train,y_train)

print("Logistic Regression Classifier score:{}".format(score))



rfc=RandomForestClassifier(n_estimators=100,criterion='entropy',random_state=0)

rfc=rfc.fit(x_train,y_train)

y_pred_rf=rfc.predict(x_test)

score1=rfc.score(x_train,y_train)

print("Random forest Classifier score:{}".format(score1))





knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

knn=knn.fit(x_train, y_train)

y_pred_knn=knn.predict(x_test)

score=knn.score(x_train,y_train)

print("K Nearest Neighbors Classifier score:{}".format(score))



nb = GaussianNB()

nb=nb.fit(x_train,y_train)

y_pred_nb=nb.predict(x_test)

score=nb.score(x_train,y_train)

print("Naive Bayes Classifier score:{}".format(score))



svc = SVC(kernel = 'rbf', random_state = 0)

svc.fit(x_train, y_train)

y_pred_svc = svc.predict(x_test)

score=svc.score(x_train,y_train)

print("Kernel SVM Classifier score:{}".format(score))





xg = XGBClassifier(max_depth=3,n_estimators=100)

xg.fit(x_train, y_train)

y_pred_xgb = xg.predict(x_test)

score=xg.score(x_train,y_train)

print("XG Boost Classifier score:{}".format(score))



#XGB Forest is the best classifier according to the prediction on test set



results=pd.DataFrame({"PassengerId":titanic_test['PassengerId'],"Survived":y_pred_xgb})

results.to_csv('Final_Output.csv',index=False)
