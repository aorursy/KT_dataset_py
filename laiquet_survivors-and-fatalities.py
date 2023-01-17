import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import pylab as pl

import warnings

from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.model_selection import train_test_split
#Disabling warnings

warnings.simplefilter('ignore')
#Importing training & testing data

data_train = pd.read_csv('../input/train.csv')

data_test = pd.read_csv('../input/test.csv')
#Dropping unrequired columns

data_train.drop(columns=['Name', 'Ticket', 'Cabin', 'Embarked'], inplace=True)

data_test.drop(columns=['Name', 'Ticket', 'Cabin', 'Embarked'], inplace=True)
#Finding empty cells

print("Training Data:\n",data_train.isna().sum(),"\n")

print("Testing Data:\n",data_test.isna().sum())
#Finding mean age & Fare

meanTrainAGE = round(data_train['Age'].mean())

meanTestAGE = round(data_test['Age'].mean())

meanTestFARE = round(data_test['Fare'].mean())
#Filling all Missing age values with mean age value

data_train['Age'].fillna(meanTrainAGE, inplace=True)

data_test['Age'].fillna(meanTestAGE, inplace=True)

data_test['Fare'].fillna(meanTestFARE, inplace=True)
#Peeking at data

print(data_train.shape)

print(data_train.describe())
data_train.head(10)
#Total No. of fatalities

plotD = data_train[data_train['Survived']==0].groupby(['Sex']).Survived.count()

pl.figure(figsize =(25,5))

pl.ylabel('Gender', fontsize=15)

pl.xlabel('Total count', fontsize=15)

pl.title('Total No. of fatalities in Males & Females', fontsize=15)

plot = plotD.plot('barh')

plot.tick_params(axis='both', which='major', labelsize=15)
#Total No. of Survivors

plotD = data_train[data_train['Survived']==1].groupby(['Sex']).Survived.count()

pl.figure(figsize =(25,5))

pl.ylabel('Gender', fontsize=15)

pl.xlabel('Total count', fontsize=15)

pl.title('Total No. of Male & Female Survivors', fontsize=15)

plot = plotD.plot('barh')

plot.tick_params(axis='both', which='major', labelsize=15)
#Data Transformation

data_train['Sex'] = np.where(data_train['Sex']=='male', 0, 1)

data_test['Sex'] = np.where(data_test['Sex']=='male', 0, 1)
#Pairplot of parameters

sns.pairplot(data_train, kind="reg")
#Correlation matrix & Heatmap

pl.figure(figsize =(10,10))

corrmat = data_train.corr()

sns.heatmap(corrmat, annot=True, fmt='.1f', vmin=0, vmax=1, square=True);
#Separating Labels and featureSet columns

columns = data_train.columns.tolist()

columns = [c for c in columns if c not in ['Survived', 'PassengerId']]

target = 'Survived'



X = data_train[columns]

y = data_train[target]
#Splitting data into training and testing sets and further normalizing training and testing FeatureSets data for better classifier's results

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)



print("Training FeatureSet:", X_train.shape)

print("Training Labels:", y_train.shape)

print("Testing FeatureSet:", X_test.shape)

print("Testing Labels:", y_test.shape)
#Initializing the model with some parameters.

model = RandomForestClassifier(n_estimators=1000, min_samples_leaf=10, random_state=1)

#Fitting the model to the data.

model.fit(X_train, y_train)

#Generating predictions for the test set.

predictions = model.predict(X_test)

#Computing the Model Accuracy

print("Random Forrest Accuracy:",metrics.accuracy_score(y_test, predictions))

#Computing the error.

print("Mean Absoulte Error:", mean_absolute_error(predictions, y_test))

#Computing classification Report

print("Classification Report:\n", classification_report(y_test, predictions))

#Plotting confusion matrix

print("Confusion Matrix:")

df = pd.DataFrame(

    confusion_matrix(y_test, predictions),

    index = [['actual', 'actual'], ['0','1']],

    columns = [['predicted', 'predicted'], ['0','1']])

print(df)
#Generating predictions for the test set.

pred_Test_Set = model.predict(data_test[columns])

pred_Test_Set
#Appending my predictions to test dataset

data_test['Survived'] = pred_Test_Set
#Test data with predictions

data_test