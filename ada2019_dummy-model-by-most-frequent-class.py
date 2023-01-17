# importing train and test data into train_df and test_df dataframes

import pandas as pd

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
# train and test dataframes column names

print(train_df.columns.values)

print('=' * 40)

print(test_df.columns.values)
# preview the train data

train_df.head(n=4)

train_df.tail(n=7)
# preview the test data

test_df.head()

train_df.tail()
# train and test features data types

train_df.info()

print('_'*40)

test_df.info()
# numerical features distribution

print(train_df.describe())

print('_'*40)

print(test_df.describe())
# categorical features distribution

print(train_df.describe(include=['O']))

print('_'*80)

print(test_df.describe(include=['O']))
# preparing training data

train_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare','Survived']

train_data = train_df[train_cols]
# retrieving 7 first rows from data frame

train_data.head(n=7)                            
# retrieving 7 last rows from data frame

train_data.tail(n=7)                            
# get amount of missing data

train_data.isnull().sum()
# massaging training data

train_data_m = train_data.copy(deep = True)     # dataframe deep copy

train_data_m = train_data_m.dropna()            # train data frame missing values imputation

x_train_data_m =  train_data_m.drop('Survived', axis = 1)                  # removing column from dataframe

y_train_data_m = train_data_m['Survived']       
# majority rule model

from sklearn.dummy import DummyClassifier

dummy_model = DummyClassifier(strategy='most_frequent')

dummy_model.fit(x_train_data_m, y_train_data_m)

dummy_model_train_prediction = dummy_model.predict(x_train_data_m) 
# estimating model accuracy on training data

from sklearn.metrics import accuracy_score

dummy_model_train_prediction_accuracy = round(accuracy_score(y_train_data_m, dummy_model_train_prediction)*100,2)

print(dummy_model_train_prediction_accuracy,'%')                              
# preparing testing data

test_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']

test_data = test_df[test_cols]

test_data.info()
# massaging testing data

test_data_m = test_data.copy(deep = True)

test_data_m['Age'].fillna((test_data_m['Age'].mean()), inplace=True)

test_data_m['Fare'].fillna((test_data_m['Fare'].mean()), inplace=True)

test_data_m.info()
# preparing submission data

ID = test_df['PassengerId']

P = dummy_model.predict(test_data_m)
# preparing submission file

submission = pd.DataFrame( { 'PassengerId': ID , 'Survived': P } )

submission.to_csv('dummy_model_v1.csv' , index = False )