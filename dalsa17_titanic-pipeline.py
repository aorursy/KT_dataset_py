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
%matplotlib inline



# Start Python Imports

import math, time, random, datetime



# Data Manipulation

import numpy as np

import pandas as pd



# Visualization 

import matplotlib.pyplot as plt

import missingno

import seaborn as sns

plt.style.use('seaborn-whitegrid')



# Preprocessing

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.metrics import mean_absolute_error 





# Machine learning

import catboost

from sklearn.model_selection import train_test_split

from sklearn import model_selection, tree, preprocessing, metrics, linear_model

from sklearn.svm import LinearSVC

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from catboost import CatBoostClassifier, Pool, cv



# Let's be rebels and ignore warnings for now

import warnings

warnings.filterwarnings('ignore')
import pandas as pd

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

Titanic_testset = pd.read_csv("../input/titanic/test.csv")

Titanic_train  = pd.read_csv("../input/titanic/train.csv")
print('Cheacking for zero values', Titanic_train.isnull().sum())

print('-'*50)

print('Checking for nanvalues', Titanic_train.isna().sum())
Titanic_train.columns
Feature = ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch',

       'Ticket', 'Fare', 'Cabin', 'Embarked']
#removing nan values and defining feature and target prediction 

y = Titanic_train.Survived  

X = Titanic_train[Feature]
#split the data into validation and traning 

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
#creating a list with all the categorical data with relatively low cardinality

categorical_data = [cate for cate in X_train_full if

                    X_train_full[cate].nunique() < 10 and 

                    X_train_full[cate].dtype == "object"]

categorical_data
#We want to find the numerical data before creating a pipline and use simpleimputer and one-hot encoding 

numerical_data = [num for num in X_train_full if X_train_full[num].dtype in ['int64', 'float64']]

numerical_data
my_cols = categorical_data + numerical_data

X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()

X_test = Titanic_testset[my_cols].copy()
len(X_train) == len(y_train)
# We will now use the pipline to get both preprocessed 



numerical_transformer = SimpleImputer(strategy = 'constant');





categorical_transformer = Pipeline(steps = [('imputer', SimpleImputer(strategy = 'most_frequent')),

                                             ('onehot', OneHotEncoder (handle_unknown = 'ignore'))

                                            ]);



preprocessor = ColumnTransformer (transformers = [('num',numerical_transformer, numerical_data), 

                                                 ('cat',categorical_transformer, categorical_data)

                                                ])
preprocessor
rep_cycle = [num for num in range(50, 450, 50)]
#optimizing model RandomForestRegressor

def RFRscore(esti_rep): 

    my_model_1 = RandomForestRegressor(n_estimators=esti_rep, random_state = 0)

    clf = Pipeline(steps=[('preprocessor', preprocessor), 

                        ('model',my_model_1)])

    clf.fit(X_train, y_train)  

    predictions_1 = clf.predict(X_valid) 

    mae_1 = mean_absolute_error(predictions_1, y_valid) 

    return mae_1



# Using 50, 100, 150, 200, 250, 300, 350, 400 in n_estimators

RFR_mae = {}

for reps in rep_cycle: 

    RFR_mae[reps] = RFRscore(reps) 

RFR_mae
#optimizing model XGBoost 



def XGBscore(esti_rep):

    my_model_2 = XGBRegressor(n_estimators=esti_rep, learning_rate = 0.01)

    clf = Pipeline(steps=[('preprocessor', preprocessor), 

                        ('model',my_model_2)])

    clf.fit(X_train, y_train)  

    predictions_2 = clf.predict(X_valid) 

    mae_2 = mean_absolute_error(predictions_2, y_valid) 

    return mae_2



# Using 50, 100, 150, 200, 250, 300, 350, 400 in n_estimators

XGB_mae = {}

for reps in rep_cycle: 

    XGB_mae[reps] = XGBscore(reps) 

XGB_mae
# We will now move on to defining our models

model_1 = RandomForestRegressor (n_estimators = min(RFR_mae, key=RFR_mae.get), random_state=0)

model_2 = XGBRegressor(n_estimators = min(XGB_mae, key=XGB_mae.get),learning_rate=0.01)
#Bundle processing 

Bundle = Pipeline(steps=[('preprocessor', preprocessor), 

                        ('model', model_2)])
#we will now use the prepocessed to to fit the model 



Bundle.fit(X_train, y_train);



#our prediction will be 



preds = Bundle.predict(X_valid)



# Mean absolute error 

mae = mean_absolute_error(preds, y_valid)

mae
#Use your trained model to generate predictions with the test data

preds_test = Bundle.predict(X_test)
print(f'Model test accuracy: {Bundle.score(X_valid, y_valid)*100:.3f}%')
#save the data to prediction file



output = pd.DataFrame({'PassengerId': gender_submission.PassengerId, 'Survived': preds_test})

output.Survived = output.Survived.astype(int)

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")