# Importing  libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

from pandas import DataFrame

%matplotlib inline
# Importing the dataset

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
df_train.info()
#correlation map

f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(df_train.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

#male vs female survive table

sns.countplot(x="Survived", hue="Sex", data=df_train, palette="Greens_d")
#age vs survive table with gender

f,ax = plt.subplots(figsize=(10, 10))

sns.boxplot(x="Survived", y="Age", hue="Sex", data=df_train, ax = ax)
sns.violinplot(x="Survived", y="Age", data=df_train, inner=None)

sns.swarmplot(x="Survived", y="Age", data=df_train, color="w", alpha=.5)
sns.pointplot(x="Sex", y="Survived", hue="Pclass", data=df_train)
sns.countplot(x = 'Embarked',hue = 'Survived', data = df_train,

            palette = 'Paired')
#looking at the data

df_train.head()
#combine test and train set

Survived = df_train['Survived']

df_train.drop('Survived', axis = 1, inplace = True)

df_combine = pd.concat([df_train, df_test], axis = 0)
#getting missing values

missing_data = df_combine.isnull()



#now we can see how much missing data there are

for column in missing_data.columns.values.tolist():

    print(column)

    print (missing_data[column].value_counts())

    print("")
#filling Embarked column that have missing values with the mode

df_combine['Embarked'].value_counts()
df_combine["Embarked"].replace(np.nan, "S", inplace = True)
#filling Age column that have missing values with the mean

mean_Age = df_combine['Age'].mean()

df_combine['Age'].replace(np.nan, mean_Age, inplace = True)
#filling Fare column that have missing values with the mean

mean_Fare = df_combine['Fare'].mean()

df_combine['Fare'].replace(np.nan, mean_Fare, inplace = True)
#dropping Cabin column

df_combine.drop('Cabin', axis = 1, inplace = True)
#dropping Ticket column

df_combine.drop('Ticket', axis = 1, inplace = True)
#info about df_combine

df_combine.info()
#looking at the data

df_combine.head()
#dropping name.We will save name columns for later

Name_Col = df_combine['Name']

df_combine.drop('Name', axis = 1, inplace = True)



#getting titles of peoples

title = []

import re

delimiters = ",", "."

regexPattern = '|'.join(map(re.escape, delimiters))

for x in Name_Col:

   x = re.split(regexPattern, x)

   title.append(x[1])



#visualize titles

df_title = pd.DataFrame(data = title)

df_title[0].unique()

f,ax2 = plt.subplots(figsize=(10, 10))

sns.countplot(x=title, palette="Greens_d", ax = ax2)
#combining SibSp and Parch columns

df_combine['family'] = df_combine['SibSp'] + df_combine['Parch']

#creating a column that contains SibSp and Parch

df_combine['hasFamily'] = np.where(df_combine['family'] > 0 , 1, 0) 

#drop the family, SibSp and Parch columns

df_combine.drop('family', axis = 1, inplace = True)

df_combine.drop('SibSp', axis = 1, inplace = True)

df_combine.drop('Parch', axis = 1, inplace = True)
#get dummy variables of categorical data

df_combine = pd.get_dummies(data = df_combine, columns = ['Sex', 'Pclass', 'Embarked'])



#drop female columns(because we don't need it)

df_combine.drop('Sex_female', axis = 1, inplace = True)



#convert uints to ints

df_combine['Sex_male'] = df_combine['Sex_male'].astype(int)

df_combine['Pclass_1'] = df_combine['Pclass_1'].astype(int)

df_combine['Pclass_2'] = df_combine['Pclass_2'].astype(int)

df_combine['Pclass_3'] = df_combine['Pclass_3'].astype(int)

df_combine['Embarked_Q'] = df_combine['Embarked_Q'].astype(int)

df_combine['Embarked_S'] = df_combine['Embarked_S'].astype(int)

df_combine['Embarked_C'] = df_combine['Embarked_C'].astype(int)



#info about data

df_combine.info()
#splitting data to train and test back

df_train = df_combine.iloc[:891]

df_test = df_combine.iloc[891:]

X_test = df_test.copy()
#get back survived column

df_train['Survived'] = Survived

#looking at data

df_train.head(10)
#getting X ve y values for modelling

X_train = df_train.iloc[:, 0:11].values

y_train = df_train.iloc[:, 11]
# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
#use models and compare



# Fitting Logistic Regression to the Training set

from sklearn.linear_model import LogisticRegression

classifierLR = LogisticRegression(random_state = 0)

classifierLR.fit(X_train, y_train)

LrScore = classifierLR.score(X_train, y_train)



# Fitting SVM to the Training set

from sklearn.svm import SVC

classifierSvm = SVC(kernel = 'rbf', random_state = 0)

classifierSvm.fit(X_train, y_train)

SvmScore = classifierSvm.score(X_train, y_train)



# Fitting Naive Bayes to the Training set

from sklearn.naive_bayes import GaussianNB

classifierNb = GaussianNB()

classifierNb.fit(X_train, y_train)

NbScore = classifierNb.score(X_train, y_train)



# Fitting XGBoost to the Training set

from xgboost import XGBClassifier

classifierXg = XGBClassifier()

classifierXg.fit(X_train, y_train)

XgScore = classifierXg.score(X_train, y_train)
#visualize for comparing models

scores_dict = {'XgBoost': XgScore, 'Naive Bayes': NbScore, 

               'Support Vector': SvmScore, 'Logistic Reg.': LrScore}

df_scores = DataFrame.from_dict(data = scores_dict, orient = 'index')

df_scores.plot.bar()
print(XgScore)
# Applying k-Fold Cross Validation

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = classifierXg, X = X_train, y = y_train, cv = 10)
accuracies.mean()
#standart deviation of accuracies

accuracies.std()
# Applying Grid Search to find the best model and the best parameters

from sklearn.model_selection import GridSearchCV

parameters = [{'max_depth': [1, 2, 3, 4, 5], 'learning_rate': [0.05, 0.1, 0.2],

               'n_estimators': [10, 100, 500, 1000], 'min_child_weight': [1, 2,

                               3, 4]},]

grid_search = GridSearchCV(estimator = classifierXg,

                           param_grid = parameters,

                           scoring = 'accuracy',

                           cv = 10,

                           n_jobs = 1)

grid_search = grid_search.fit(X_train, y_train)

best_accuracy = grid_search.best_score_

best_parameters = grid_search.best_params_
best_accuracy
#Fitting and Predicting

grid_search.best_estimator_.fit(X_train, y_train)

y_pred = grid_search.best_estimator_.predict(X_test)
#create submission.csv

Submission = pd.DataFrame({'PassengerId' : df_test['PassengerId'],

                           'Survived' : y_pred })

Submission.head(10)
sns.countplot(x="Survived", data=Submission, palette="husl")

Submission.to_csv("Submission.csv", index = False)