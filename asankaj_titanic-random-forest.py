# Importing the libraries



# remove warnings

import warnings

warnings.filterwarnings('ignore')

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

sns.set_style('whitegrid')



# Importing the dataset

dataset = pd.read_csv('../input/train.csv')

testSet=pd.read_csv('../input/test.csv')

print("data set info","*"*40)

dataset.info()



print("test set info","*"*40)

testSet.info()

print("*"*40)





#descriptive statistics

#distribution of numerical feature values across the samples

dataset.describe()

print("*"*40)



list(dataset) #get column namess 



#get details about catagorical variables

dataset.describe(include=['O'])



#Select the columns to use in the model 

dataset = dataset.iloc[:,[0,1,2,4,5,9]]

testSet = testSet.iloc[:,[0,1,3,4,8]]





#X must be a data set to view this 

print(dataset['PassengerId'].isnull().sum())

print(dataset['Pclass'].isnull().sum())

print(dataset['Sex'].isnull().sum())

print(dataset['Age'].isnull().sum())

print(dataset['Fare'].isnull().sum())



print("*"*40)

#test set check null 

print(testSet['PassengerId'].isnull().sum())

print(testSet['Pclass'].isnull().sum())

print(testSet['Sex'].isnull().sum())

print(testSet['Age'].isnull().sum())

print(testSet['Fare'].isnull().sum())



# Taking care of missing data in Age

from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values="NaN", strategy = 'mean', axis = 0)

imputer = imputer.fit(dataset.iloc[:,4:5].values)

dataset.iloc[:,4:5]= imputer.transform(dataset.iloc[:,4:5].values)



imputer2 = Imputer(missing_values="NaN", strategy = 'mean', axis = 0)

imputer = imputer2.fit(testSet.iloc[:,3:5].values)

testSet.iloc[:,3:5]= imputer2.transform(testSet.iloc[:,3:5].values)





# Encoding categorical data

from sklearn.preprocessing import LabelEncoder

labelencoder_X = LabelEncoder()

dataset.iloc[:,3:4] = labelencoder_X.fit_transform(dataset.iloc[:,3:4].values)





labelencoder_X1 = LabelEncoder()

testSet.iloc[:,2:3] = labelencoder_X1.fit_transform(testSet.iloc[:,2:3].values)



X = dataset.iloc[:,[2,3,4,5]].values

y = dataset.iloc[:,1].values





#observe how survival reflects with the variables

#No survivours vs Age 

gan = sns.FacetGrid(dataset, col='Survived')

gan.map(plt.hist, 'Age', bins=10)



#No survivours vs Age and class

grid = sns.FacetGrid(dataset, col='Survived', row='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();





list(dataset)



#No survivors vs Sex 



gan = sns.FacetGrid(dataset, col='Survived')

gan.map(plt.hist, 'Sex', bins=10)



gan = sns.FacetGrid(dataset, col='Survived')

gan.map(plt.hist, 'Age', bins=10)
list(dataset)


"""# Splitting the dataset into the Training set and Test set

#from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0, random_state = 0)

"""

X_train=X

y_train=y

X_test=testSet.iloc[:,1:].values





 #Fitting Random Forest Regression to the Training set

from sklearn.ensemble import RandomForestRegressor

Regressor = RandomForestRegressor(n_estimators = 100,oob_score=True, random_state = 0)

Regressor.fit(X_train, y_train)





# Check the importance of variables

Regressor.feature_importances_

F_importance=pd.Series(Regressor.feature_importances_,index=(dataset.iloc[:,[2,3,4,5]]).columns)

F_importance.plot(kind="barh" , figsize=(7,6))









 #Fitting Random Forest Classification to the Training set

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators =195, criterion = 'entropy', random_state = 0)

classifier.fit(X_train, y_train)



y_pred = classifier.predict(X_test)



"""# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)"""



se



# Applying k-Fold Cross Validation

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)

accuracies.mean()

print("Accuracy =",accuracies.mean())





#grid search hyper parameter tuning

"""

# Applying Grid Search to find the best model and the best parameters

from sklearn.model_selection import GridSearchCV

parameters = [{'n_estimators': [195,196,197]}]

grid_search = GridSearchCV(estimator = classifier,

                           param_grid = parameters,

                           scoring = 'accuracy',

                           cv = 10,

                           n_jobs = -1)

grid_search = grid_search.fit(X_train, y_train)

best_accuracy = grid_search.best_score_

best_parameters = grid_search.best_params_



got accuracy of 0.830212234707for n_estimator=195

"""





outPut =pd.read_csv('../input/test.csv').iloc[:,[0]]

se = pd.Series(y_pred)

outPut["Survive"]=se.values



print(outPut)

#outPut.to_csv('../input/output.csv', sep='\t', encoding='utf-8')