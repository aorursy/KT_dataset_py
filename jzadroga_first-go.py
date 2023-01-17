# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pylab as plot
import keras
from keras.models import Sequential
from keras.layers import Dense

def drop_columns(csv):
    #drop columns that are not needed for now
    dropped_csv = csv.drop(columns=['PassengerId', 'Fare', 'Name', 'SibSp', 'Parch', 'Ticket', 'Embarked'])
    return dropped_csv

def process_sex_col(data_file):
    #convert sex to 0 or 1
    data_file['Sex'] = data_file['Sex'].map({'male':1, 'female':0})
    return data_file

def process_age_col(data_file):
    #fil in missing age
    data_file['Age'] = np.around(data_file['Age'].fillna(data_file['Age'].median()))
    return data_file

def process_fare_col(data_file):
    #fil in missing age
    data_file['Fare'] = data_file['Fare'].fillna(data_file['Fare'].median())
    return data_file

def process_pclass_col(data_file):
    #parse the data out into seperate column categories
    pclass_dummies = pd.get_dummies(data_file['Pclass'], prefix="Pclass")
    data_file = pd.concat([data_file, pclass_dummies], axis=1)
    
    #now remove Pclass
    data_file.drop('Pclass', axis=1, inplace=True)
    return data_file

def process_cabin_col(data_file):
    data_file.loc[data_file['Cabin'].notnull(), 'Cabin'] = 1
    data_file['Cabin'] = data_file['Cabin'].fillna(0)
    print(data_file.head)
    return data_file
    
training_data = drop_columns(pd.read_csv('../input/train.csv'))
training_data = process_sex_col(training_data)
training_data = process_age_col(training_data)
#training_data = process_fare_col(training_data)
training_data = process_pclass_col(training_data)
training_data = process_cabin_col(training_data)

X = training_data.drop('Survived', axis=1).values
Y = training_data[['Survived']].values

# Define the model
model = Sequential()
model.add(Dense(50, input_dim=6, activation='relu', name='layer_1'))
#model.add(Dense(25, activation='relu', name='layer_2'))
#model.add(Dense(50, activation='relu', name='layer_3'))
model.add(Dense(1, activation='linear', name='output_layer'))
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(
    X,
    Y,
    epochs=50,
    shuffle=True,
    verbose=2
)

# Load the separate test data set
test_data = pd.read_csv("../input/test.csv")

test_data_passengers = test_data[['PassengerId']].values

test_data = drop_columns(test_data)
test_data = process_sex_col(test_data)
test_data = process_age_col(test_data)
#test_data = process_fare_col(test_data)
test_data = process_pclass_col(test_data)
test_data = process_cabin_col(test_data)

test_values = test_data.values

#Make a prediction with the neural network
prediction = model.predict(test_values)

#convert the data and submit
columns=['PassengerId','Survived']
num = np.around(prediction)
num = num.astype(int)

results = np.append(test_data_passengers, num, axis=1)
my_submission = pd.DataFrame(results, columns=columns)
my_submission.to_csv('submission.csv', index=False)

# Any results you write to the current directory are saved as output.

#