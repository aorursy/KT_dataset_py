# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import glmnet_py

import statsmodels

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

from sklearn.metrics import roc_auc_score, make_scorer

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

import glmnet_python 

from glmnet import glmnet

import glmnet_python

from glmnet import glmnet

from glmnetPlot import glmnetPlot

from glmnetPrint import glmnetPrint; from glmnetCoef import glmnetCoef; from glmnetPredict import glmnetPredict

from cvglmnet import cvglmnet; from cvglmnetCoef import cvglmnetCoef

from cvglmnetPlot import cvglmnetPlot; from cvglmnetPredict import cvglmnetPredict

import scipy

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
titanic_df = pd.read_csv('/kaggle/input/titanic.csv', sep='\t')
titanic_df.head()
titanic_df.shape
count_survived = titanic_df[titanic_df['Survived']==1].shape[0]

count_not_survived = titanic_df[titanic_df['Survived']==0].shape[0]

print('Number of people who survived: ', count_survived)

print('Number of people who did not survive: ', count_not_survived)
drop_indices = np.random.choice(titanic_df[titanic_df['Survived']==1].index, 40, replace=False)

print(drop_indices)
titanic_df = titanic_df.drop(drop_indices)
count_survived = titanic_df[titanic_df['Survived']==1].shape[0]

count_not_survived = titanic_df[titanic_df['Survived']==0].shape[0]

print('Number of people who survived: ', count_survived)

print('Number of people who did not survive: ', count_not_survived)
#check for any other unusable values

print(pd.isnull(titanic_df).sum())
titanic_df["CabinBool"] = (titanic_df["Cabin"].notnull().astype('int'))
titanic_df = titanic_df.drop(['Cabin'], axis = 1)

titanic_df= titanic_df.drop(['Ticket'], axis = 1)
#now we need to fill in the missing values in the Embarked feature

print("Number of people embarking in Southampton (S):")

southampton = titanic_df[titanic_df["Embarked"] == "S"].shape[0]

print(southampton)



print("Number of people embarking in Cherbourg (C):")

cherbourg = titanic_df[titanic_df["Embarked"] == "C"].shape[0]

print(cherbourg)



print("Number of people embarking in Queenstown (Q):")

queenstown = titanic_df[titanic_df["Embarked"] == "Q"].shape[0]

print(queenstown)
#replacing the missing values in the Embarked feature with S

titanic_df = titanic_df.fillna({"Embarked": "S"})

titanic_df = titanic_df.fillna(round(titanic_df['Age'].mean(), 2))
#check for any other unusable values

print(pd.isnull(titanic_df).sum())
titanic_df = titanic_df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'CabinBool']]
titanic_df['Pclass'] = titanic_df['Pclass'].astype('category')

titanic_df['SibSp'] = titanic_df['SibSp'].astype('category')

titanic_df['Parch'] = titanic_df['Parch'].astype('category')

titanic_df['Embarked'] = titanic_df['Embarked'].astype('category')

titanic_df['CabinBool'] = titanic_df['CabinBool'].astype('category')
titanic_df = pd.get_dummies(titanic_df, prefix=['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'CabinBool'], drop_first=True)
target = titanic_df['Survived']

predictors = titanic_df[titanic_df.columns.difference(['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'CabinBool'])]
predictors.columns
print(type(target))

print(type(predictors))
#check for any other unusable values

print(pd.isnull(titanic_df).sum())
X = titanic_df.loc[:, titanic_df.columns != 'Survived']

y = titanic_df['Survived']
print(y)
from imblearn.over_sampling import SMOTE



smote = SMOTE(ratio='minority')

X_sm, y_sm = smote.fit_sample(X, y)
y_sm
count_survived = list(y_sm).count(1)

count_not_survived = list(y_sm).count(0)

print('Number of people who survived: ', count_survived)

print('Number of people who did not survive: ', count_not_survived)
from sklearn.utils import shuffle

X_sm, y_sm = shuffle(X_sm, y_sm,random_state=0)
y_sm
ordered_cv = TimeSeriesSplit(n_splits=3)
alpha_values = [0, 0.5, 1]

lambda_values = scipy.float64([0.01, 0.05, 0.5])

for alpha_val in alpha_values:

    for lambda_val in lambda_values:

        AUC_values = []

        print("-------------------------------------------------")

        print("alpha = " + str(alpha_val))

        print("lambda = " + str(lambda_val))

        for train_index, test_index in ordered_cv.split(X_sm):

            #print("TRAIN:", train_index, "TEST:", test_index)

            X_train, X_test = X_sm[train_index], X_sm[test_index]

            y_train, y_test = y_sm[train_index], y_sm[test_index]

  

            fit = glmnet(

                x = X_train.astype(scipy.float64),

                y = y_train.astype(scipy.float64),

                family = 'binomial',

                alpha = alpha_val,

                standardize = True)

            prediction = glmnetPredict(

                fit, 

                newx = X_test.astype(scipy.float64),

                s = scipy.float64([lambda_val]), 

                ptype = 'class')

            AUC = roc_auc_score(y_test, prediction)

            print("cv sample AUC: " + str(AUC))

            AUC_values.append(AUC)

        #print(AUC_values)

        mean_auc = sum(AUC_values) / len(AUC_values)

        print("AUC avg score: "+ str(mean_auc))
X_np = X.to_numpy()

y_np = y.to_numpy()

alpha_values = [0, 0.5, 1]

lambda_values = scipy.float64([0.01, 0.05, 0.5])

for alpha_val in alpha_values:

    for lambda_val in lambda_values:

        AUC_values = []

        print("-------------------------------------------------")

        print("alpha = " + str(alpha_val))

        print("lambda = " + str(lambda_val))

        for train_index, test_index in ordered_cv.split(X_np):

#             print("TRAIN:", train_index, "TEST:", test_index)

            X_train, X_test = X_np[train_index], X_np[test_index]

            y_train, y_test = y_np[train_index], y_np[test_index]

  

            fit = glmnet(

                x = X_train.astype(scipy.float64),

                y = y_train.astype(scipy.float64),

                family = 'binomial',

                alpha = alpha_val,

                standardize = True)

            prediction = glmnetPredict(

                fit, 

                newx = X_test.astype(scipy.float64),

                s = scipy.float64([lambda_val]), 

                ptype = 'class')

            AUC = roc_auc_score(y_test, prediction)

            print("cv sample AUC: " + str(AUC))

            AUC_values.append(AUC)

        #print(AUC_values)

        mean_auc = sum(AUC_values) / len(AUC_values)

        print("AUC avg score: "+ str(mean_auc))
# Construct the pipeline with a standard scaler and a logistic regression

estimators = []

estimators.append(('standardize', StandardScaler()))

estimators.append(('LogisticRegression', LogisticRegression(solver='liblinear')))

model = Pipeline(estimators)
penalty_lst = ['l1', 'l2']

C_lst = [0.001,0.01,0.1,1]

logistic_regression = LogisticRegression(solver='liblinear')

parameters = {

    'LogisticRegression__penalty': penalty_lst,

    'LogisticRegression__C': C_lst

}



clf = GridSearchCV(

    estimator=model,

    param_grid=parameters,

    cv=ordered_cv,

    scoring=make_scorer(roc_auc_score))

clf = clf.fit(X_sm, y_sm)

best_estimator = clf.best_estimator_
best_estimator
best_score = clf.best_score_

print(best_score)