# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestClassifier

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.graph_objects as go

import plotly.express as px

from xgboost import XGBClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
ds_train = pd.read_csv("/kaggle/input/titanic/train.csv")

ds_test = pd.read_csv("/kaggle/input/titanic/test.csv")

ds_result = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")

print('Dataset is imported')
print('Training set has {} datapoints'.format(ds_train.shape[(0)]))

print('Test set has {} datapoints'.format(ds_test.shape[(0)]))
ds_train.describe()
ds_train.info()
ds_train.head()
#drop the Ticket and cabin column for simplicity

ds_train = ds_train.drop(["Ticket", "Cabin", "Name"],axis = 1)

ds_train.head()
#visualizing the age distribution

age_sr = pd.Series(ds_train['Age'].value_counts())

fig = px.scatter(age_sr, y= age_sr.values, x = age_sr.index)

fig.show()
print('No. of passengers having age less than 20 is {}'.format(len(ds_train[ds_train['Age']<20])))

#drop the Embarked column for simplicity

ds_train = ds_train.drop(["Embarked"],axis = 1)

ds_train.head()
ds_train.corr()
ds_train['Family_size'] = ds_train['SibSp'] +ds_train['Parch'] + 1

ds_train.head()
age_by_Pclass = ds_train.groupby(['Sex', 'Pclass']).median()['Age']

for Pclass in range(1,4):

    for Sex in ['male', 'female']:

        print('Median of pclass{} of {}s :{}'.format(Pclass, Sex, age_by_Pclass[Sex][Pclass]))
ds_train['Age']= ds_train.groupby(['Sex','Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))

ds_train.info()
#Replacing 'Male' and 'Female' with '0' and '1' respectively

ds_train=ds_train.replace(to_replace='male',value=0)

ds_train=ds_train.replace(to_replace='female',value=1)

ds_train.head()
X_train = ds_train.drop(['Survived'], axis =1)

y_train = ds_train['Survived'].values

print('x_train shape: {}'.format(x_train.shape))

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
#Filling missing fare with median fare

null_index=ds_test['Fare'].isnull().index

medianFare=ds_test['Fare'].median()

ds_test.at[null_index,'Fare'] = medianFare

print("Missing Fare updated as Median Fare :{}".format(medianFare))
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

#y_pred_xgb=classifier_xgb.predict(X_test)



#Converting 2 dimensional  y_pred array into single dimension 

y_pred_rf=y_pred_rf.ravel()

#y_pred_xgb=y_pred_xgb.ravel()



#Creating submission data frame and subsequent csv file for submission

submission_df_rf = pd.DataFrame(columns=['PassengerId', 'Survived'])

submission_df_rf['PassengerId'] = X_test['PassengerId'].astype(int)

submission_df_rf['Survived'] = y_pred_rf

submission_df_rf.to_csv('submissions_rf.csv', header=True, index=False)



# submission_df_xgb = pd.DataFrame(columns=['PassengerId', 'Survived'])

# submission_df_xgb['PassengerId'] = X_test['PassengerId'].astype(int)

# submission_df_xgb['Survived'] = y_pred_xgb

# submission_df_xgb.to_csv('submissions_xgb.csv', header=True, index=False)