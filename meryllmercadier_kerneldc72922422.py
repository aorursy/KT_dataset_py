# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
import numpy as np

import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler

from sklearn.impute import SimpleImputer, KNNImputer

from sklearn.compose import make_column_transformer

import pandas as pd





frames = [train_data,test_data]

full_data = pd.concat(frames)

master_data = full_data

print(train_data.shape)

print(test_data.shape)

print(full_data.shape)



#Deleting all the columns that doesn't seem relevant

full_data = full_data.drop(['PassengerId','Survived','Ticket','Cabin'],axis = 1)

print(full_data.shape)



#Only keeping the titles in the "Name" column, to create an interesting new feature

full_data['Name'] = full_data['Name'].str.rsplit(',').str[-1]

full_data['Name'] = full_data['Name'].str.split('.').str[0]

#Regrouping titles

full_data['Name'] = full_data['Name'].replace(['Lady','the Countess','Capt', 'Col',

    'Dona','Don', 'Dr', 'Major','','Rev', 'Sir', 'Jonkheer'], 'Rare',regex=True)

full_data['Name'] = full_data['Name'].replace('Mlle', 'Miss', regex=True)

full_data['Name'] = full_data['Name'].replace('Ms', 'Miss',regex=True)

full_data['Name'] = full_data['Name'].replace('Mme', 'Mrs',regex=True)



#Checking the result

pd.crosstab(full_data['Name'], full_data['Sex'])



#Deleting all the rows with missing age

full_data.dropna(subset = ["Age"], inplace=True)



#Checking if i forget any empty value in my dataset

print(np.where(pd.isnull(full_data)))



#Deleting the raws containing NaN values

full_data.dropna(how='any', inplace=True)

#Checking if it really worked

print(np.where(pd.isnull(full_data)))



#Targeting the "age" to build a model to predict missing ages

y = full_data['Age']

X = full_data.drop(['Age'],axis = 1)



#Spliting the dataset for futur train and test

from sklearn.model_selection import train_test_split

X_train ,X_test ,y_train ,y_test = train_test_split(X ,y ,test_size=0.2,random_state = 14)



#Encoding features and using GradientBoosting Regressor



numerical_features = ['SibSp','Parch','Fare']

categorical_features = ['Sex','Embarked','Pclass','Name']



numerical_pipeline = make_pipeline(MinMaxScaler())

categorical_pipeline = make_pipeline(OneHotEncoder())





preprocessor = make_column_transformer((numerical_pipeline, numerical_features),(categorical_pipeline, categorical_features))



from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

#Pipeline

model = make_pipeline(preprocessor, GradientBoostingRegressor())



model.fit(X_train, y_train)



#Using a grid search to found the best model's parameters(in our case, this is estimators : 2000 and learning rate : 0.01 )

from sklearn.model_selection import GridSearchCV

params = {'gradientboostingregressor__n_estimators' : [2000]  

         , 'gradientboostingregressor__learning_rate' : [0.01]}

grid = GridSearchCV(model, param_grid= params, cv=5)

grid.fit(X_train, y_train)

print(grid.best_params_)

print('train score :' , grid.best_score_)

best_model = grid.best_estimator_

print('test score :', best_model.score(X_test,y_test))



#Making the train_database to make the Survive prediction model

train_data.dropna(subset = ["Fare","Embarked"], inplace=True)

train_data = train_data.drop(['Ticket','Cabin'], axis = 1)

print(train_data.shape)

#Only keeping the titles in the "Name" column, to create an interesting new feature

train_data['Name'] = train_data['Name'].str.rsplit(',').str[-1]

train_data['Name'] = train_data['Name'].str.split('.').str[0]

#Regrouping titles

train_data['Name'] = train_data['Name'].replace(['Lady','the Countess','Capt','Col','Dona','Don', 'Dr','Major','','Rev','Sir','Jonkheer'], 'Rare',regex=True)

train_data['Name'] = train_data['Name'].replace('Mlle', 'Miss', regex=True)

train_data['Name'] = train_data['Name'].replace('Ms', 'Miss',regex=True)

train_data['Name'] = train_data['Name'].replace('Mme', 'Mrs',regex=True)

train_data



#Integrating Predicted age

train_data['age_predicted'] = best_model.predict(train_data)

train_data['Age'].fillna(train_data['age_predicted'], inplace=True)

train_data = train_data.drop(['age_predicted'],axis = 1)

train_data



#Creating a family size feature

train_data['family_size'] = train_data['SibSp'] + train_data['Parch']+1

train_data



#Splitting age by categories (to reduce the error risk from previous prediction's score)

split = [0, 3, 14, 24, 34, 44, 54, 64]

names = ['0', '1', '2', '3', '4','5','6','7']

d = dict(enumerate(names, 1))

train_data['Age'] = np.vectorize(d.get)(np.digitize(train_data['Age'], split))

train_data



#Targeting "survived" to build a survive prediction model

master_y = train_data['Survived']

master_X = train_data.drop(['Survived'], axis = 1)

print('master_y',master_y.shape)

print('master_X',master_X.shape)



master_X_train ,master_X_test ,master_y_train ,master_y_test = train_test_split(master_X ,master_y ,test_size=0.2,random_state = 14)



##Encoding features and using GradientBoosting Classifier



master_numerical_features = ['SibSp','Parch','Fare']

master_categorical_features = ['Sex','Embarked','Pclass','Name','family_size']



master_numerical_pipeline = make_pipeline(MinMaxScaler())

master_categorical_pipeline = make_pipeline(OneHotEncoder())





master_preprocessor = make_column_transformer((master_numerical_pipeline, master_numerical_features),(master_categorical_pipeline, master_categorical_features))



from sklearn.ensemble import GradientBoostingClassifier

#Pipeline

master_model = make_pipeline(master_preprocessor, GradientBoostingClassifier())



master_model.fit(master_X_train, master_y_train)

master_model.fit(master_X_test, master_y_test)



#Using a grid search to found the best model's parameters(in our case, this is estimators : 1000 and learning rate : 0.01 )

master_params = {'gradientboostingclassifier__n_estimators' : [1000]  

         , 'gradientboostingclassifier__learning_rate' : [0.01]}

master_grid = GridSearchCV(master_model, param_grid= master_params, cv=5)

master_grid.fit(master_X_train, master_y_train)

print(master_grid.best_params_)

print('Master train score :' , master_grid.best_score_)

master_best_model = master_grid.best_estimator_

print('Master test score :', master_best_model.score(master_X_test,master_y_test))



#Building the test_database

test_data = test_data.drop(['Ticket','Cabin'], axis = 1)

test_data['Fare'].fillna(value = test_data['Fare'].median(), inplace = True)



#Only keeping the titles in the "Name" column, to create an interesting new feature

test_data['Name'] = test_data['Name'].str.rsplit(',').str[-1]

test_data['Name'] = test_data['Name'].str.split('.').str[0]

#Regrouping titles

test_data['Name'] = test_data['Name'].replace(['Lady','the Countess','Capt','Col','Dona','Don', 'Dr','Major','','Rev','Sir','Jonkheer'], 'Rare',regex=True)

test_data['Name'] = test_data['Name'].replace('Mlle', 'Miss', regex=True)

test_data['Name'] = test_data['Name'].replace('Ms', 'Miss',regex=True)

test_data['Name'] = test_data['Name'].replace('Mme', 'Mrs',regex=True)



#Integrating Predicted age

test_data['age_predicted'] = best_model.predict(test_data)

test_data['Age'].fillna(test_data['age_predicted'], inplace=True)

test_data = test_data.drop(['age_predicted'],axis = 1)

#Splitting age by categories (to reduce the error risk from previous prediction )

split = [0, 3, 14, 24, 34, 44, 54, 64]

names = ['0', '1', '2', '3', '4','5','6','7']

d = dict(enumerate(names, 1))

test_data['Age'] = np.vectorize(d.get)(np.digitize(test_data['Age'], split))

test_data



#Creating a family size feature

test_data['family_size'] = test_data['SibSp'] + test_data['Parch']+1

test_data



predictions = master_best_model.predict(test_data).astype(int)

predictions



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")
