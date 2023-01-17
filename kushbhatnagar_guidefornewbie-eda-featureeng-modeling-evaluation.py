# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Importing required libraries

#Importing the required libraries and data set 

import numpy as np

import seaborn as sns

import pandas as pd

import matplotlib.pyplot as plt

import plotly.graph_objects as go

import plotly.express as px

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier



print("Important libraries loaded successfully")
ds_train=pd.read_csv("/kaggle/input/titanic/train.csv")

ds_test=pd.read_csv("/kaggle/input/titanic/test.csv")

ds_result=pd.read_csv("/kaggle/input/titanic/gender_submission.csv")

print("Train and Test data sets are imported successfully")
print("Test and Training data details are as follows: ")

print('Number of Training Examples = {}'.format(ds_train.shape[0]))

print('Number of Test Examples = {}\n'.format(ds_test.shape[0]))
ds_train.head()
#Drop columns from training data set

ds_train=ds_train.drop(['Ticket','Cabin'],axis=1)

print("Columns Dropped Successfully")

ds_train.head()
#Converting Age into series and visualizing the age distribution

age_series=pd.Series(ds_train['Age'].value_counts())

fig=px.scatter(age_series,y=age_series.values,x=age_series.index)

fig.update_layout(

    title="Age Distribution",

    xaxis_title="Age in Years",

    yaxis_title="Count of People",

    font=dict(

        family="Courier New, monospace",

        size=18,

    )

)

fig.show()
print("Number of teenagers and child passengers in ship are {}".format(len(ds_train[ds_train['Age'] < 20 ])))
print("Number of Passengers Gender Wise \n{}".format(ds_train['Sex'].value_counts()))

#Gender wise distribution

fig = go.Figure(data=[go.Pie(labels=ds_train['Sex'],hole=.4)])

fig.update_layout(

    title="Sex Distribution",

    font=dict(

        family="Courier New, monospace",

        size=18

    ))

fig.show()
#Create categorical variable graph for Age,Sex and Survived variables

sns.catplot(x="Survived", y="Age", hue="Sex", kind="swarm", data=ds_train,height=10,aspect=1.5)

plt.title('Passengers Survival Distribution: Age and Sex',size=25)

plt.show()

#Visualize relation between Pclass and Survival

fig = go.Figure(data=[go.Pie(labels=ds_train['Pclass'],hole=.4)])

fig.update_layout(

    title="PClass Distribution",

    font=dict(

        family="Courier New, monospace",

        size=18

    ))

fig.show()
#Visualize PClass and Survival

#Create categorical variable graph for Age,Pclass and Survived variables

sns.catplot(x="Survived", y="Age", hue="Pclass", kind="swarm", data=ds_train,height=10,aspect=1.5)

plt.title('Passengers Survival Distribution: Age and Pclass',size=25)

plt.show()
#Visualize Fare and Survival

#Create categorical variable graph for Sex,Fare and Survived variables

sns.catplot(x="Survived", y="Fare", hue="Sex", kind="swarm", data=ds_train,height=8,aspect=1.5)

plt.title('Passengers Survival Distribution: Fare and Sex',size=20)

plt.show()
#Visualize relation between Embarked and Survival

fig = go.Figure(data=[go.Pie(labels=ds_train['Embarked'],hole=.4)])

fig.update_layout(

    title="Embarked Distribution",

    font=dict(

        family="Courier New, monospace",

        size=18

    ))

fig.show()
#Visualize Embarked and Survival

#Create categorical variable graph for Embarked,Age and Survived variables

sns.catplot(x="Survived", y="Age", hue="Embarked", kind="swarm", data=ds_train,height=8,aspect=1.5)

plt.title('Passengers Survival Distribution: Embarked and Age',size=20)

plt.show()
#Drop columns from training data set

ds_train=ds_train.drop(['Embarked','Name'],axis=1)

print("Columns Dropped Successfully")

ds_train.head()
# Training set high correlations

ds_train.corr()
#Add new column 'Family Size' in training model set

ds_train['Family_Size'] = ds_train['SibSp'] + ds_train['Parch'] + 1

print("Family Size column created sucessfully")

ds_train.head()
#Visualize Family size and Survival

sns.barplot(x="Family_Size", y="Age", hue="Survived", data=ds_train,palette = 'rainbow')

plt.title('Family Size - Age Survival Distribution',size=20)

plt.show()

sns.catplot(y="Family_Size", x="Survived", hue='Sex',kind="swarm", data=ds_train,height=8,aspect=1.5)

plt.title('Family Size - Gender Survival Distribution',size=20)

plt.show()
print("Information on Train Data Set :")

ds_train.info()
age_by_pclass_sex = ds_train.groupby(['Sex', 'Pclass']).median()['Age']



for pclass in range(1, 4):

    for sex in ['female', 'male']:

        print('Median age of Pclass {} {}s: {}'.format(pclass, sex, age_by_pclass_sex[sex][pclass]))

print('Median age of all passengers: {}'.format(ds_train['Age'].median()))



# Filling the missing values in Age with the medians of Sex and Pclass groups

ds_train['Age'] = ds_train.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))
print("Information on Train Data Set :")

ds_train.info()
#Replacing 'Male' and 'Female' with '0' and '1' respectively

ds_train=ds_train.replace(to_replace='male',value=0)

ds_train=ds_train.replace(to_replace='female',value=1)

ds_train.head()
X_train=ds_train.drop(['Survived'],axis=1)

y_train=ds_train['Survived'].values

print('X_train shape: {}'.format(X_train.shape))

print('y_train shape: {}'.format(y_train.shape))
classifier_rf=RandomForestClassifier(criterion='gini', 

                                           n_estimators=1100,

                                           max_depth=5,

                                           min_samples_split=4,

                                           min_samples_leaf=5,

                                           max_features='auto',

                                           oob_score=True,

                                           random_state=42,

                                           n_jobs=-1,

                                           verbose=1)

classifier_rf.fit(X_train,y_train)
classifier_xgb=XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)

classifier_xgb.fit(X_train,y_train)
ds_test.info()
age_by_pclass_sex = ds_test.groupby(['Sex', 'Pclass']).median()['Age']



for pclass in range(1, 4):

    for sex in ['female', 'male']:

        print('Median age of Pclass {} {}s: {}'.format(pclass, sex, age_by_pclass_sex[sex][pclass]))

print('Median age of all passengers: {}'.format(ds_test['Age'].median()))



# Filling the missing values in Age with the medians of Sex and Pclass groups

ds_test['Age'] = ds_test.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))
ds_test.info()
#Filling missing fare with median fare

null_index=ds_test['Fare'].isnull().index

medianFare=ds_test['Fare'].median()

ds_test.at[null_index,'Fare'] = medianFare

print("Missing Fare updated as Median Fare :{}".format(medianFare))
ds_test.info()
#Drop columns from test data set

ds_test=ds_test.drop(['Ticket','Cabin','Embarked','Name'],axis=1)

print("Columns Dropped Successfully")



#Creating Family Size columns from test data set

ds_test['Family_Size'] = ds_test['SibSp'] + ds_test['Parch'] + 1

print("Family Size column created sucessfully")



#Encoding Gender column from test data set

ds_test=ds_test.replace(to_replace='male',value=0)

ds_test=ds_test.replace(to_replace='female',value=1)

X_test=ds_test



X_test.head()
#Prediction test results

y_pred_rf=classifier_rf.predict(X_test)

y_pred_xgb=classifier_xgb.predict(X_test)



#Converting 2 dimensional  y_pred array into single dimension 

y_pred_rf=y_pred_rf.ravel()

y_pred_xgb=y_pred_xgb.ravel()



#Creating submission data frame and subsequent csv file for submission

submission_df_rf = pd.DataFrame(columns=['PassengerId', 'Survived'])

submission_df_rf['PassengerId'] = X_test['PassengerId'].astype(int)

submission_df_rf['Survived'] = y_pred_rf

submission_df_rf.to_csv('submissions_rf.csv', header=True, index=False)



submission_df_xgb = pd.DataFrame(columns=['PassengerId', 'Survived'])

submission_df_xgb['PassengerId'] = X_test['PassengerId'].astype(int)

submission_df_xgb['Survived'] = y_pred_xgb

submission_df_xgb.to_csv('submissions_xgb.csv', header=True, index=False)
#Apply K-fold in current model to check model accuracy

from sklearn.model_selection import cross_val_score

accuracies_rf = cross_val_score(estimator = classifier_rf, X = X_train, y = y_train, cv = 10)

accuracies_xgb = cross_val_score(estimator = classifier_xgb, X = X_train, y = y_train, cv = 10)
#Checking accuracies for 10 fold in Random Forest and XG Boost Models

print("Accuracies for 10 Fold in Random Forest Model is {}".format(accuracies_rf))

print("Accuracies for 10 Fold in XG Boost Model is {}".format(accuracies_xgb))
#Checking Mean and Standard Deviation between Accuracies

print("Mean Accuracy for Random Forest Model is {}".format(accuracies_rf.mean()))

print("Mean Accuracy for XG Boost Model is {}".format(accuracies_xgb.mean()))

print("Standard Deviation for Random Forest Model is {}".format(accuracies_rf.std()))

print("Standard Deviation for XG Boost Model is {}".format(accuracies_xgb.std()))
#Importing required library for Grid Search

from sklearn.model_selection import GridSearchCV

#Create the parameter grid based on the results of random search

param_grid = { 'bootstrap': [True],

              'max_depth': [80, 90, 100, 110],

              'max_features': [2, 3], 

              'min_samples_leaf': [3, 4, 5],

              'min_samples_split': [8, 10, 12],

              'n_estimators': [100, 300, 500, 1000] }

grid_search = GridSearchCV(estimator = classifier_rf, param_grid = param_grid,cv = 3, n_jobs = -1) 

grid_search = grid_search.fit(X_train, y_train)
#Getting the best params

best_accuracy = grid_search.best_score_

best_parameters = grid_search.best_params_

print("Best Accuracy for Random Forest Classifier is {}".format(best_accuracy))

print("Best Parameters for Random Forest Classifier is {}".format(best_parameters))
#Creating new classifier and fitting Training set

classifier_rf_new = RandomForestClassifier(n_estimators = 719,

                                           bootstrap=False,

                                           max_depth=464,

                                           max_features=0.3,

                                           min_samples_leaf=1,

                                           min_samples_split=2,

                                           random_state=42)

classifier_rf_new.fit(X_train, y_train)
print("Predicting Results from new Classifier and Converting into Submission file")

# Predicting the Train set results

y_pred_rf_new=classifier_rf_new.predict(X_test)

#Converting 2 dimensional  y_pred array into single dimension 

y_pred_rf_new=y_pred_rf_new.ravel()

#Creating submission data frame and subsequent csv file for submission

submission_df_rf_new = pd.DataFrame(columns=['PassengerId', 'Survived'])

submission_df_rf_new['PassengerId'] = X_test['PassengerId'].astype(int)

submission_df_rf_new['Survived'] = y_pred_rf_new

submission_df_rf_new.to_csv('submissions_rf_new.csv', header=True, index=False)

print("Created Submission file from new classifier successfully")