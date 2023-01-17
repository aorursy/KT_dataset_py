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
#define the path for the data

path = '/kaggle/input/titanic/'



#inspect gender_submission.csv - convert to dataframe and see how the submission needs to look

sub_df = pd.read_csv(path + 'gender_submission.csv')



#create train and test data frames

train_df = pd.read_csv(path + 'train.csv')

test_df = pd.read_csv(path + 'test.csv')

#inspect how we need to submission

sub_df.head()
#We need the passnger id and then whether or not the passengers survived

#Next we will do some exploratory data analysis

train_df.shape
#so 12 columns of data to explore

train_df.columns
#We will look at the head using .head()

train_df.head()
#we will reorder the columns for surviced is at the end

train_df = train_df[['PassengerId', 'Name', 'Age', 'Sex', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin',

                   'Embarked', 'Survived']]
#ensure it's moved where we want it

train_df.head()
test_df = test_df[['PassengerId', 'Name', 'Age', 'Sex', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin',

                   'Embarked']]
test_df = test_df.drop(['Name', 'Ticket', 'Cabin'], axis = 1)
#to make things easy we can drop the Name column, otherwise you can look for promient titles, Mr., Miss, Dr, etc

train_df = train_df.drop('Name', axis = 1)
#we will check to makre sure it's dropped

train_df.head()
#now we will check for NaN values

train_df.isnull().sum()
#out of the 891 rows, 687 don't have a cabin entry so we will drop it as well

train_df = train_df.drop('Cabin', axis = 1)
#And inspect to ensure the dataframe is what we expect

train_df.head()
#embarked only had 2 NaN, so we will find the most used value and use those

train_df['Embarked'].value_counts()
train_df['Embarked'] = train_df['Embarked'].fillna('S')
test_df['Embarked'] = test_df['Embarked'].fillna('S')
#Now we have to deal with age

#Let's do some data analysis

train_df['Age'].value_counts()
train_df['Age'].mode()
train_df['Age'].median()
train_df['Age'].mean()
#still not there yet on age, but we'll move forward for now

#we will create some dummy variables for the categorical features (another way is to use CatBoost)

Sex_dummy = pd.get_dummies(train_df['Sex'])

Embark_dummy = pd.get_dummies(train_df['Embarked'])



#We'll add them to the data frame and then drop the original columns

train_df = pd.concat([train_df, Sex_dummy, Embark_dummy], axis = 1)

Sex_dummy = pd.get_dummies(test_df['Sex'])

Embark_dummy = pd.get_dummies(test_df['Embarked'])

test_df = pd.concat([test_df, Sex_dummy, Embark_dummy], axis = 1)

test_df = test_df.drop(['Sex', 'Embarked'], axis = 1)
#let's drop the columns and then inspect again

train_df = train_df.drop(['Sex', 'Embarked'], axis = 1)
train_df.head()
#let's look at the Ticket

train_df.groupby('Ticket')['Fare'].value_counts().plot()
#it doesn't look like anything big, so let's drop it for now

train_df = train_df.drop(['Ticket'], axis = 1)
train_df.head()
#let reorder to surviced is once again at the end

train_df = train_df[['PassengerId', 'Age', 'SibSp', 'Parch', 'Fare', 'female',

                    'male', 'C', 'Q', 'S', 'Survived']]
train_df.head()
#let's check on class balance, ie Survived to Not Surviced

train_df['Survived'].value_counts()
#It's slightly off, so we will try to use some balancing in our model

#this is a classification problem and we will be using XGBOOST, which can handle the NaNs in Age

#first we will use SciKitLearn and split the data
import xgboost as xgb

from sklearn.metrics import mean_squared_error

import pandas as pd

import numpy as np
#Seperate the target class

X, y = train_df.iloc[:,:-1],train_df.iloc[:,-1]

#create the DMatrix

data_dmatrix = xgb.DMatrix(data=X,label=y)
#split the data, mainly for metric purposes

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
#construct the model and hyperparamets

xg_cl = xgb.XGBClassifier(objective ='reg:logistic', colsample_bytree = 0.3, learning_rate = 0.001,

                max_depth = 2, alpha = 10, n_estimators = 10)
#we will build the actual model

xg_cl.fit(X_train,y_train)

#let's get our predictions

preds = xg_cl.predict(X_test)

#finally we'll see how we did

rmse = np.sqrt(mean_squared_error(y_test, preds))

print("RMSE: %f" % (rmse))
#lets make the test set look right and then run it through the model for submission

preds = xg_cl.predict(test_df)
submission = pd.DataFrame()

submission['PassengerId'] = test_df['PassengerId']

submission['Survived'] = xg_cl.predict(test_df)

submission.to_csv('/kaggle/working/submission.csv', index = False)
#next we will look at using the Catboost Algorithmn

#since catboost can handle categorical features we will first reset our train and test dataframes

print(train_df.shape)

print(test_df.shape)
#the test set is about 50% of the training set, so we probably won't overfit

train_df.isnull().sum()
#cabin looks like it offers nothing to the model since it contains so many NaN values, therefore we will drop it from the test and training sets

#we will look at the unique values of the other features and see if any have dominate features

#train_df['Embarked'].value_counts()

    

#PassengerId is just a unique number like an index and it doesnt offer much, so we will drop it, it's required for submission, so well save it, but it won't be part of the model

#Pclass has a good distribution without any dominate features

#Sex has a good distribution

#Age has can be better modeled with some feature engineering

#SibSp dominates with 600 equal to 0, so we will drop that

#Parch dominates with 678 equal to 0, so we will drop that

#Ticket are fairly unique so we will keep them for now

#Fare has a many varied values so weill keep that

#Cabin was previously dropped

#Embarked has 72% equal to one class, so we will drop that as well

train_df = train_df.drop(['SibSp','Embarked', 'Parch', 'Cabin'] , axis = 1)

test_df = test_df.drop(['SibSp','Embarked', 'Parch', 'Cabin'] , axis = 1)





train_df.head()
#need to use the catboost classificaiton algorithimg

import catboost as ctb

from catboost import CatBoostClassifier
X = train_df.drop('Survived', axis = 1)

y = train_df['Survived']
#split the data, mainly for metric purposes

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
cat_features = np.where(X.dtypes != float)[0]
model = CatBoostClassifier(

    iterations = 50000,

    learning_rate = .001,

    loss_function = 'Logloss',

    early_stopping_rounds = 20000)
model.fit(X_train, y_train, cat_features = cat_features, eval_set = (X_test, y_test),plot = True)
submission_2 = pd.DataFrame()

submission_2['PassengerId'] = test_df['PassengerId']

submission_2['Survived'] = model.predict(test_df)

submission_2['Survived'] = submission_2['Survived'].astype(int)

submission_2.to_csv('submission_2.csv', index = False)
from catboost import Pool

test_pool = Pool(X_test, y_test, cat_features)

model.eval_metrics(test_pool, 'Accuracy')