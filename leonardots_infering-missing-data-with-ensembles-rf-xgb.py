import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import sklearn.neighbors as knn

import sklearn.preprocessing

import sklearn.ensemble



from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score



import os

print(os.listdir("../input"))
#reading of data and EDA

titanic = pd.read_csv('../input/train.csv')

print(titanic.info())



titanic.plot(kind='box',subplots=True,figsize=(20,5))

plt.show()
#taking a better look into the non-numeric values

print('=========Sex=========')

print(titanic['Sex'].describe())

print(titanic['Sex'].unique())



print('=========TICKET=========')

print(titanic['Ticket'].describe())

print(titanic['Ticket'].unique())



print('=========CABIN=========')

print(titanic['Cabin'].describe())

print(titanic['Cabin'].unique())



print('=========EMBARKED=========')

print(titanic['Embarked'].describe())

print(titanic['Embarked'].unique())
#cleaning of the data

#changing sex from string to a binary type (1 = female and 0 = male)

titanic['Sex'] = titanic['Sex'].apply(lambda x: 1 if x == 'female' else 0)



#using the information from https://www.kaggle.com/gunesevitan/advanced-feature-engineering-tutorial-with-titanic we are going to assign value "S" to the missing embarked features

titanic['Embarked']=titanic['Embarked'].fillna('S')



#we are going to one hot encode the embarked feature

titanic = titanic.join(pd.get_dummies(titanic['Embarked']).rename(columns={'C':'Cherbourg','Q':'Queenstown','S':'Southampton'}))



#for the tickets, we are just going to remove the whole column as it doesn't seem to add much to the problem

titanic = titanic.drop('Ticket',axis='columns')



#last check before age infering

print(titanic.info())
#we are going to use random forest for infering the missing age

corr = titanic.dropna().corr()

fig, ax = plt.subplots(figsize=(20, 10))

sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, ax =ax)
#as seen in the heatmap, we are going to use Pclass, SibSp and Parch to infer the data

cleanTitanic = titanic[['Pclass','Sex','SibSp','Parch','Age']].dropna()

ageTrainingFeatures = cleanTitanic[['Pclass','Sex','SibSp','Parch']].astype('float')

ageTrainingValues = cleanTitanic['Age']



#uniformizing the data

scaler = sklearn.preprocessing.MinMaxScaler()

ageTrainingFeatures = pd.DataFrame(scaler.fit_transform(ageTrainingFeatures))



#testing the values with cross validation

X_train, X_test, y_train, y_test = train_test_split(ageTrainingFeatures, ageTrainingValues, test_size=0.2, random_state=0)



rfAge = sklearn.ensemble.RandomForestRegressor(n_estimators=15,max_depth=5,random_state=0)

rfAge.fit(X_train,y_train)



print(rfAge.score(X_test,y_test))
#having our model, we are now going to predict the values for the missing ages

nullAgesMask = titanic['Age'].isnull()

missingAges = titanic[nullAgesMask]



titanic.loc[nullAgesMask,'Age'] = rfAge.predict(missingAgesFeatures)



print(titanic.info())
#using as base the information from https://www.kaggle.com/gunesevitan/advanced-feature-engineering-tutorial-with-titanic, we are going to infer the missing cabins

#diferently from Güneş Evitan, we are going to one hot encode the cabins' "class" and infer the missing ones



listOfCabins = list(titanic['Cabin'].dropna().str[0].unique())

listOfCabins.sort()

listOfCabins = listOfCabins[:-1] #removing the "T" cabin

print(listOfCabins)



#we are going to fill the dataframe

cabinsStringList = titanic['Cabin']

#print(cabinsStringList)

for i,cabin in enumerate(cabinsStringList):

    if (type(cabin)==str):

        titanic.loc[i,'Cabin'] = listOfCabins.index(cabin.replace('T','A')[0])   

        

titanic['Cabin'] = titanic['Cabin'].astype('float')

print(titanic['Cabin'].describe())
#each column of the one hot encode is going to be the target of our inference

#but first, lets see the correlation between the other columns and the cabins

basicFeatures = ['Pclass','Age','SibSp','Parch','Fare','Sex','Cherbourg','Queenstown','Southampton']

corr = titanic[basicFeatures+['Cabin']].dropna().corr()

fig, ax = plt.subplots(figsize=(20, 10))

sns.heatmap(corr, annot=True, ax = ax)
#as expected, 'SibSp' and 'Parch' have little to no impact on whether a person is in a cabin or not, as well as if it departed from 'Queenstown'

#we are now going to use random forest classifier to infer the cabins

cleanTitanic = titanic[['Pclass','Age','Fare','Sex','Cherbourg','Southampton','Cabin']].dropna()



cabinTrainingFeatures = cleanTitanic.drop('Cabin',axis='columns').astype('float')

cabinTrainingValues = cleanTitanic['Cabin']

    

#testing the values with cross validation

X_train, X_test, y_train, y_test = train_test_split(cabinTrainingFeatures, cabinTrainingValues, test_size=0.2, random_state=0)



rfCabin = sklearn.ensemble.RandomForestClassifier(n_estimators=15,max_depth=5, random_state=0)

rfCabin.fit(X_train,y_train)



print(rfCabin.score(X_test,y_test))
#with the random forest dict in hands, we are going to fill the missing cabins in the titanic dataframe

nullCabinMask = titanic['Cabin'].isnull()

missingCabin = titanic[nullCabinMask]



missingCabinFeatures = missingCabin[['Pclass','Age','Fare','Sex','Cherbourg','Southampton']].astype('float')

titanic.loc[nullCabinMask,'Cabin'] = rfCabin.predict(missingCabinFeatures)
#with the data cleaned and with all the values needed, we are going to train a random forest for predicting the survivors

#we are going to use random forest for infering the missing age values giving weight to the most correlated variables

corr = titanic.corr().abs()

corr['Survived'].drop('Survived',axis='rows').sort_values(ascending=False).plot(kind='bar')

plt.show()



survivedTraining = titanic.drop(['Name','PassengerId','Embarked'],axis='columns')
#training of the random forest

print(survivedTraining.info())

survivedTrainingFeatures = survivedTraining.loc[:,survivedTraining.columns != 'Survived']

survivedTrainingValues = survivedTraining['Survived']



#testing the values with cross validation

X_train, X_test, y_train, y_test = train_test_split(survivedTrainingFeatures, survivedTrainingValues, test_size=0.3, random_state=0)



rfSurvived = sklearn.ensemble.RandomForestClassifier(n_estimators=20,max_depth=7, random_state=0)

rfSurvived.fit(X_train,y_train)



print(rfSurvived.score(X_test,y_test))

#cross validation test

scores = cross_val_score(rfSurvived, survivedTrainingFeatures, survivedTrainingValues, cv=5)

print(scores)
import xgboost as xgb



data_dmatrix = xgb.DMatrix(data=survivedTrainingFeatures,label=survivedTrainingValues)



#cross_val

n_folds = 10

early_stopping = 10

params = {'eta': 0.02, 'max_depth': 4, 'subsample': 0.7, 'colsample_bytree': 0.7, 'objective': 'binary:logistic', 'seed': 99, 'silent': 1, 'eval_metric':'auc', 'nthread':4}



cv = xgb.cv(params, data_dmatrix, 5000, nfold=n_folds, early_stopping_rounds=early_stopping, verbose_eval=0)



#doing XGBoost's training

booster = xgb.train(params, data_dmatrix, num_boost_round = cv.index[-1], verbose_eval=1)

predictions = np.array(booster.predict(data_dmatrix))

print('Accuracy: '+str((predictions.round()==survivedTrainingValues.values).sum()/predictions.shape[0]))
#loading the test set

titanic = pd.read_csv('../input/test.csv')

print(titanic.info())



titanic.plot(kind='box',subplots=True,figsize=(20,5))

plt.show()
#cleaning of the data

#changing sex from string to a binary type (1 = female and 0 = male)

titanic['Sex'] = titanic['Sex'].apply(lambda x: 1 if x == 'female' else 0)



#using the information from https://www.kaggle.com/gunesevitan/advanced-feature-engineering-tutorial-with-titanic we are going to assign value "S" to the missing embarked features

titanic['Embarked']=titanic['Embarked'].fillna('S')



#we are going to one hot encode the embarked feature

titanic = titanic.join(pd.get_dummies(titanic['Embarked']).rename(columns={'C':'Cherbourg','Q':'Queenstown','S':'Southampton'}))



#for the tickets, we are just going to remove the whole column as it doesn't seem to add much to the problem

titanic = titanic.drop('Ticket',axis='columns')



#last check before age infering

print(titanic.info())
#having our model, we are now going to predict the values for the missing ages

nullAgesMask = titanic['Age'].isnull()

missingAges = titanic[nullAgesMask]



missingAgesFeatures = missingAges[['Pclass','Sex','SibSp','Parch']].astype('float')

titanic.loc[nullAgesMask,'Age'] = rfAge.predict(missingAgesFeatures)



print(titanic.info())
#using knn to infer the missing fare

corr = titanic.corr()

fig, ax = plt.subplots(figsize=(20, 10))

sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, ax =ax)
#every feature seems important for fare, so we will use them all (aside from passengerId)

maskTitanic = [x for x in titanic.columns if x not in survivedTraining.columns]

maskSurvived = [x for x in survivedTraining.columns if x not in titanic.columns]



knnTrainingData = pd.concat([titanic.drop(maskTitanic,axis='columns'),survivedTraining.drop(maskSurvived,axis='columns')]).dropna()



#we are going to use all features to infer the data

fareTrainingFeatures = knnTrainingData.loc[:,knnTrainingData.columns != 'Fare'].drop('Cabin',axis='columns')

fareTrainingValues = knnTrainingData['Fare']



#testing the values with cross validation

X_train, X_test, y_train, y_test = train_test_split(fareTrainingFeatures, fareTrainingValues, test_size=0.2, random_state=0)



rfFare = sklearn.ensemble.RandomForestRegressor(n_estimators=15,max_depth=5,random_state=0)

rfFare.fit(X_train,y_train)



print(rfFare.score(X_test,y_test))



#with the random forest in hands, we are going to fill the missing fare in the titanic dataframe

nullFareMask = titanic['Fare'].isnull()

missingFare = titanic[nullFareMask]



missingFareFeatures = missingFare[[x for x in titanic.columns if x in fareTrainingFeatures.columns]]



#having our model, we are now going to predict the values for the missing ages

titanic.loc[nullFareMask,'Fare'] = rfFare.predict(missingFareFeatures)



print(titanic.info())
#completing the one hot encoding of the available cabins

#now we are going to create a DataFrame for hold the one hot encoding of the cabins

cabinsStringList = titanic['Cabin']



for i,cabin in enumerate(cabinsStringList):

    if (type(cabin)==str):

        titanic.loc[i,'Cabin'] = listOfCabins.index(cabin.replace('T','A')[0])   

        

titanic['Cabin'] = titanic['Cabin'].astype('float') 



#with the random forest dict in hands, we are going to fill the missing cabins in the titanic dataframe

nullCabinMask = titanic['Cabin'].isnull()

missingCabin = titanic[nullCabinMask]



missingCabinFeatures = missingCabin[['Pclass','Age','Fare','Sex','Cherbourg','Southampton']].astype('float')

titanic.loc[nullCabinMask,'Cabin'] = rfCabin.predict(missingCabinFeatures)

    

print(titanic.info())
survivedTest = xgb.DMatrix(titanic.drop(['Name','PassengerId','Embarked'],axis='columns'))

predictions = np.array(booster.predict(survivedTest))



predictionsCSV = pd.DataFrame()

predictionsCSV['PassengerId'] = titanic['PassengerId']

#predictionsCSV['SurvivedXGB'] = predictions

#predictionsCSV['SurvivedRF'] = rfSurvived.predict(titanic.drop(['Name','PassengerId','Cabin','Embarked'],axis='columns'))

#predictionsCSV['Survived'] = ((predictionsCSV['SurvivedRF']+predictionsCSV['SurvivedXGB'])/2).round().astype('int')

predictionsCSV['Survived'] = predictions.round().astype('int')
#predictionsCSV.drop(['SurvivedRF','SurvivedXGB'],axis='columns').to_csv('predictions.csv',index=False)

predictionsCSV.to_csv('predictions.csv',index=False)