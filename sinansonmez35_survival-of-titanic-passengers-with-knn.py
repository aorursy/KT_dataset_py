import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
import itertools
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics

warnings.filterwarnings('ignore')

# Open the data
df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")
df_test_copy = df_test.copy()
df_train_copy = df_train.copy()
# print(df_train.columns)
# print(df_train.describe())
# print(df_train.shape)

# check info for each column to see which lines are empty and then we will process the data
# train data age, cabin and embarked has some missing data
print(df_train.info())

print('-' * 30)
# test data age, fare, cabin and embarked has some missing data
print(df_test.info())
# calculate average age by passenger class to fill empty age cells
print('Age breakdown by passenger class: ')
print(df_train.groupby('Pclass').mean()[['Age']])

# Fill empty values in age column
df_train['Age'].fillna(df_train['Age'].median(), inplace=True)
df_test['Age'].fillna(df_test['Age'].median(), inplace=True)

# fill empty data in embarked and fare column
df_train['Embarked'].fillna(df_train['Embarked'].mode()[0], inplace=True)
df_test['Fare'].fillna(df_test['Fare'].median(), inplace=True)

# check out rows if there are any empty cell left in Age column
print('Number of empty cells in train data: ')
print(df_train.isnull().sum())
print('-' * 30)
print('Number of empty cells in test data: ')
print(df_test.isnull().sum())

# drop Cabin, passenger ID and Ticket value from train and test data
drop_columns = ['Cabin', 'PassengerId', 'Ticket', 'Name']
df_train = df_train.drop(drop_columns, axis=1)
df_test = df_test.drop(drop_columns, axis=1)

# change non-numerical value to numerical values
df_train['Sex'] = df_train['Sex'].map({'female': 0, 'male': 1}).astype(int)
df_test['Sex'] = df_test['Sex'].map({'female': 0, 'male': 1}).astype(int)

df_train['Embarked'] = df_train['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
df_test['Embarked'] = df_test['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

#check if non numerical values are converted
print(df_train.head())
# correlation matrix
plt.figure(figsize=(14, 12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(df_train.astype(float).corr(), linewidths=0.1, vmax=1.0,
            square=True, cmap="BuPu", linecolor='white', annot=True)
# create scatter plot to check sex, Pclass, fare and embarked relationship with survived
plt.figure(figsize=(10, 5))
sns.boxplot(x='Survived', y='Age', data=df_train)

# mean of ages by Survived
print(df_train.groupby('Survived').mean()[['Age']])

plt.figure(figsize=(10, 5))
sns.boxplot(x='Survived', y='Fare', data=df_train)

# mean of ages by Survived
print(df_train.groupby('Survived').mean()[['Fare']])
# as you may remember I changed female/male value to 0 and 1 to create correlation matrix.
# I reversed that action to create the plot and reversed again. This seems like repetitive task. Please advise how to improve my code
df_train['Sex'] = df_train['Sex'].map({0: 'female', 1: 'male'}).astype(str)
sns.countplot(x='Sex', data=df_train, hue='Survived')
df_train['Sex'] = df_train['Sex'].map({'female': 0, 'male': 1}).astype(int)
df_train['Embarked'] = df_train['Embarked'].map({0: 'S', 1: 'C', 2: 'Q'}).astype(str)
sns.countplot(x='Embarked', data=df_train, hue='Survived')
df_train['Embarked'] = df_train['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
features = ['Age', 'Fare', 'Sex', 'Embarked']

# split train and test data
X = df_train[features]

# Normalize Data
X = preprocessing.StandardScaler().fit_transform(X)

y = df_train['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
# try 3 different Ks to pick the best one to train the model
neigh5 = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
yhat5 = neigh5.predict(X_test)
cm5 = metrics.confusion_matrix(y_test, yhat5)
print('Test set Accuracy of K=5:', metrics.accuracy_score(y_test, yhat5))
print(cm5)

neigh17 = KNeighborsClassifier(n_neighbors=17).fit(X_train, y_train)
yhat17 = neigh17.predict(X_test)
cm17 = metrics.confusion_matrix(y_test, yhat17)
print('Test set Accuracy of K=17: ', metrics.accuracy_score(y_test, yhat17))
print(cm17)


neigh29 = KNeighborsClassifier(n_neighbors=29).fit(X_train, y_train)
yhat29 = neigh29.predict(X_test)
cm29 = metrics.confusion_matrix(y_test, yhat29)
print('Test set Accuracy of K=29 : ', metrics.accuracy_score(y_test, yhat29))
print(cm29)
# use model to predict on test data and submit the file
X_submit = np.array(df_test[features])
y_submit = neigh29.predict(X_submit)

submit = df_test_copy[['PassengerId']].copy()
submit['Survived'] = y_submit

print('Length of an survived value array: ', len(y_submit))
print(submit.head())

submit.to_csv('result_knn_titanic.csv', index=False)
