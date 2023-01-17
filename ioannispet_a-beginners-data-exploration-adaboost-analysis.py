#A Beginners data exploration on data classification

#Feel free to comment any input i might missed or anything that could improve this kernel





# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Store the data in a dataframe

df = pd.read_csv('/kaggle/input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')
#Print head

df.head()
#Get the number of rows and columns

df.shape
#Count the missing values

df.isna().sum()
#View statistics

df.describe()
#count the number of employees that left or stayed

df['Attrition'].value_counts()
#Visualize the above number

plt.figure(figsize=(10, 6))

sns.countplot(df['Attrition'])
#If the model guessed no all the times whats the possibility to be correct

#The model has to beat this number

no_guessing = round((1233 - 237) / 1233, 3)

print(no_guessing)
#Show the employee attrition by age

plt.figure(figsize=(18, 8))

sns.countplot(x='Age', hue='Attrition', data=df)
#Remove some useless columns

df = df.drop('Over18', axis=1)

df = df.drop('EmployeeNumber', axis=1)

df = df.drop('StandardHours', axis=1)

df = df.drop('EmployeeCount', axis=1)
#Get the correlation

df.corr()
#Visualize the correlation

plt.figure(figsize=(25, 10))

sns.heatmap(df.corr(), annot=True, fmt='.0%')
#Transform non numerical to numerical

for column in df.columns:

    if df[column].dtype == np.number:

        continue

    else:

        df[column] = LabelEncoder().fit_transform(df[column])
#Bring attrition column in the first Position

cols = list(df.columns.values)

#cols

df = df[['Attrition',

 'Age',

 'BusinessTravel',

 'DailyRate',

 'Department',

 'DistanceFromHome',

 'Education',

 'EducationField',

 'EnvironmentSatisfaction',

 'Gender',

 'HourlyRate',

 'JobInvolvement',

 'JobLevel',

 'JobRole',

 'JobSatisfaction',

 'MaritalStatus',

 'MonthlyIncome',

 'MonthlyRate',

 'NumCompaniesWorked',

 'OverTime',

 'PercentSalaryHike',

 'PerformanceRating',

 'RelationshipSatisfaction',

 'StockOptionLevel',

 'TotalWorkingYears',

 'TrainingTimesLastYear',

 'WorkLifeBalance',

 'YearsAtCompany',

 'YearsInCurrentRole',

 'YearsSinceLastPromotion',

 'YearsWithCurrManager']]
#Split the data

X = df.iloc[:, 1:df.shape[1]]

Y = df.iloc[:, 0]
#Split the data ( 75% training and 25% testing )

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
#Lets find out the best params setup for a balanced model

n_estimators_range = list(range(10, 200, 10))

learning_rate_range = np.arange(0.1, 2, 0.1)

algorithm_options = ['SAMME', 'SAMME.R']



ABC = AdaBoostClassifier(random_state=0)



param_grid = dict(n_estimators=n_estimators_range, learning_rate=learning_rate_range, algorithm=algorithm_options)



grid = GridSearchCV(ABC, param_grid, scoring='roc_auc', n_jobs=-1)

grid.fit(X_train, Y_train)
#Show the scores

print('The best params are: {}'.format(grid.best_params_))

print()

print('The best score is: {}'.format(grid.best_score_))

print()

print('The mean cross-validated score is: {}'.format(grid.score(X_test, Y_test)))
#Use AdaBoost

forest = AdaBoostClassifier(n_estimators=130, learning_rate=1.3, random_state=0, algorithm='SAMME')

forest.fit(X_train, Y_train)
#Get the accuracy on the training data set

round(forest.score(X_train, Y_train), 3)
#Show the classification report

prediction = forest.predict(X_test)

target_names = ['TP + FP', 'TN + FN']

print(classification_report(Y_test, prediction, target_names=target_names))
#Show the confusion matrix and accuracy score for the model on the test data

cm = confusion_matrix(Y_test, forest.predict(X_test))

TP = cm[0][0]

FP = cm[0][1]

FN = cm[1][0]

TN = cm[1][1]

print()

print(cm)

print()

print('True positive is: {}'.format(TP))

print('False positive is: {}'.format(FP))

print('False negative is: {}'.format(FN))

print('True negative is: {}'.format(TN))
print('Model beeing correct guessing only no = {}'.format(no_guessing))

print('Model Testing Accuracy = {}'.format(round(forest.score(X_test, Y_test), 3)))
#Show the prediction score ( The ability of the model not to predict attrit for the employees that actually wont attrit)

round(precision_score(Y_test, prediction), 3)
#Show the recall score (What percentage of employees that end up attriting does the model succesfully find )

round(recall_score(Y_test, prediction), 3)
#Show the f1 score

round(f1_score(Y_test, prediction), 3)
#Inspect which feature contributes more to attrition

feat_importances = pd.Series(forest.feature_importances_, index=X.columns)

plt.figure(figsize=(20, 10))

feat_importances.nlargest(20).plot(kind='barh')
#As we see from the above plot the 2 variables that contribute more is 1: Monthly income, 2: Total working years





#A Beginners data exploration on data classification

#Feel free to comment any input i might missed or anything that could improve this kernel