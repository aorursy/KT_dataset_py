#Importing Data Analysis Libs

import pandas as pd

import numpy as np

import warnings

warnings.filterwarnings('ignore')
#Getting .csv files 

df = pd.read_csv('../input/train.csv')

dfTest = pd.read_csv('../input/test.csv')
#Checking the first lines

df.head()
#Data Types

df.dtypes
#Checking the shape of the dataframe

df.shape
dfTest.head(3)
dfTest.shape
#Statistic Summary

df.describe()
dfTest.describe()
#Cheking columns with null values in Dataframe

df.isnull().any()
# Pclass column distribution

df.groupby('Pclass').size()
#Copying Dataframe 

dfT = df
#Checking Unique values from Sex Column

sex_values = df.drop_duplicates('Sex')

print(sex_values['Sex'])
#Label Encoding for Sex Column

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

le.fit(dfT['Sex'])

#list(le.classes_)

dfT['Sex'] = le.transform(dfT['Sex'])





le.fit(dfTest['Sex'])

#list(le.classes_)

dfTest['Sex'] = le.transform(dfTest['Sex'])
#Checking Unique values from Embarked Column

embarked_values = df.drop_duplicates('Embarked')

print(embarked_values['Embarked'])
#Using mode to fulfill null values on Embarked Column

dfT['Embarked'].fillna(dfT['Embarked'].mode()[0], inplace=True)

dfTest['Embarked'].fillna(dfTest['Embarked'].mode()[0], inplace=True)
#Label Encoding for Embarked Column

le.fit(dfT['Embarked'])

#list(le.classes_)

dfT['Embarked'] = le.transform(dfT['Embarked'])



le.fit(dfTest['Embarked'])

#list(le.classes_)

dfTest['Embarked'] = le.transform(dfTest['Embarked'])
#Using Mean to fulfill null values on Column Age

dfT['Age'].fillna(dfT['Age'].mean(), inplace=True)

dfTest['Age'].fillna(dfTest['Age'].mean(), inplace=True)
#Dataset Formatting

dfT.shape
dfT.dtypes
#Statistic Summary

dfT.describe()
#Checking nulls on Dataframe

dfT.isnull().any()
dfTest.isnull().any()
#Completing column Fare with 0 instead of Null

dfTest['Fare'] = dfTest['Fare'].fillna(0)
#Visualization libs

#import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
# Histogram

dfT.hist()

plt.show()
# Correlation Matrix with names

columns = ['pID', 'surviv', 'pclass', 'sex','age', 'sibsp', 'parch', 'fare', 'Emb']

correlations = dfT.corr()

print (correlations)
# Plot

#import numpy as np

fig = plt.figure()

ax = fig.add_subplot(111)

ax.set_color_cycle(['red', 'black', 'yellow'])

cax = ax.matshow(correlations, interpolation='nearest', vmin = -1, vmax = 1)

#cax = ax.imshow(correlations, interpolation='nearest', vmin = -1, vmax = 1)

fig.colorbar(cax)

ticks = np.arange(0, 10, 1)

ax.set_xticks(ticks)

ax.set_yticks(ticks)

ax.set_xticklabels(columns)

ax.set_yticklabels(columns)

plt.show()
#Adding columns SibSp and Parch => FamilySize

dfT['FamilySize'] = dfT['SibSp'] + dfT['Parch']

dfTest['FamilySize'] = dfTest['SibSp'] + dfTest['Parch']
new_columns = ['pID', 'surviv', 'pclass', 'sex','age', 'Fam', 'fare', 'Emb']
# New Plot

fig = plt.figure()

ax = fig.add_subplot(111)

ax.set_color_cycle(['red', 'black', 'yellow'])

cax = ax.matshow(correlations, interpolation='nearest', vmin = -1, vmax = 1)

fig.colorbar(cax)

ticks = np.arange(0, 10, 1)

ax.set_xticks(ticks)

ax.set_yticks(ticks)

ax.set_xticklabels(new_columns)

ax.set_yticklabels(new_columns)

plt.show()
#Visualizing data with seaborn

import seaborn as sns
#Dropping name columns (Name, Cabin and Ticket) from Dataframe

dfT = dfT.drop(['Name','Cabin','Ticket'],axis=1)

dfTest = dfTest.drop(['Name','Cabin','Ticket'],axis=1)
dfT.dtypes
dfTest.dtypes
# Pairplot   ====> Must have all columns without nulls

sns.pairplot(dfT)  
# kdeplot

sns.kdeplot(dfT)
dfT.head(3)
# Putting Data in the same scale (between 0 and 1)



# Importing libraries

#from pandas import read_csv

from sklearn.preprocessing import MinMaxScaler



colTrain = ['PassengerId', 'Pclass', 'Sex','Age', 'Fare', 'Embarked','FamilySize', 'Survived']

dfMLTrain = dfT[colTrain]

arrayTrain = dfMLTrain.values



colTest = ['PassengerId', 'Pclass', 'Sex','Age', 'Fare', 'Embarked','FamilySize']

dfMLTest = dfTest[colTest]

arrayTest = dfMLTest.values



# Splitting array in input and output

XTrain = arrayTrain[:,0:7]

YTrain = arrayTrain[:,7]

XTest = arrayTest[:,0:7]



# Creating new scale

scaler = MinMaxScaler(feature_range = (0, 1))

rescaledXTrain = scaler.fit_transform(XTrain)

rescaledXTest = scaler.fit_transform(XTest)





# Data transformed

print(rescaledXTrain[0:5,:])
# Feature Selection using chi2 test



# Import modules

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2



# Selecting the 5 better features that can be used in prediction model

test = SelectKBest(score_func = chi2, k = 5) 

fit = test.fit(XTrain, YTrain)



# Summarizing score

print(fit.scores_)

features = fit.transform(XTrain)



# Summarizing selected Features

print(features[0:5,:])
# Importing libraries

from sklearn import model_selection

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC





# Defining number of folds

num_folds = 10

num_instances = len(XTrain)

seed = 7



# Preparing models

modelos = []

modelos.append(('LR', LogisticRegression()))

modelos.append(('LDA', LinearDiscriminantAnalysis()))

modelos.append(('NB', GaussianNB()))

modelos.append(('KNN', KNeighborsClassifier()))

modelos.append(('CART', DecisionTreeClassifier()))

modelos.append(('SVM', SVC()))



# Model Evaluation

resultados = []

nomes = []



for nome, modelo in modelos:

    kfold = model_selection.KFold(n_splits = num_folds, random_state = seed)

    cv_results = model_selection.cross_val_score(modelo, XTrain, YTrain, cv = kfold, scoring = 'accuracy')

    resultados.append(cv_results)

    nomes.append(nome)

    msg = "%s: %f (%f)" % (nome, cv_results.mean(), cv_results.std())

    print(msg)



# Boxplot to compare algorithms

fig = plt.figure()

fig.suptitle('Comparison of Classification Algorithms')

ax = fig.add_subplot(111)

plt.boxplot(resultados)

ax.set_xticklabels(nomes)

plt.show()
# Creating logistic regression model 

modelLR = LogisticRegression()



# Training model and checking the score

modelLR.fit(XTrain, YTrain)

modelLR.score(XTrain, YTrain)



# Colecting coefficients

print('Coefficient: \n', modelLR.coef_)

print('Intercept: \n', modelLR.intercept_)



# Predictions

YPred = modelLR.predict(XTest)
#Checking accuracy

acc_log = round(modelLR.score(XTrain, YTrain) * 100, 2)

acc_log
from sklearn.ensemble import GradientBoostingClassifier



ModelCLF = GradientBoostingClassifier(n_estimators = 650, learning_rate = 1.0, max_depth = 1, random_state = 0)



# Training model and checking the score

ModelCLF.fit(XTrain, YTrain)

ModelCLF.score(XTrain,YTrain)  



# Predictions

YPredGBC=ModelCLF.predict(XTest)
#Checking accuracy

acc_log = round(ModelCLF.score(XTrain,YTrain) * 100, 2)

acc_log
survived = YPredGBC.astype(int)
#Creating Submission file

submission = pd.DataFrame({

        "PassengerId": dfTest["PassengerId"],

        "Survived": YPredGBC

    })



#submission.to_csv('../output/submission.csv', index=False)