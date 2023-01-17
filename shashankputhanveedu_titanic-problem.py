# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler

import keras
from keras.models import Sequential
from keras.layers import Dense

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Load the data using pandas
train_data_df = pd.read_csv("/kaggle/input/titanic/train.csv")
train_data_df.head()
# Drop the columns that we dont want to use
train_data = train_data_df.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1)
# Perform mean imputation for NAs in Age and fill NAs in Embarked with 'S' as that is the most frequent value/level in the variable
train_data[['Age']] = train_data[['Age']].fillna(value=29)
train_data[['Embarked']] = train_data[['Embarked']].fillna(value='S')
# Inspect the Training data
train_data.info()
# Put the data into two arrays to separate out dependent and independent variables.
X_train = train_data.iloc[:,1:9].values
Y_train = train_data.iloc[:, 0]
# Use labelencoder to convert the levels into integer encoded values.
labelencoder_X_1 = LabelEncoder()
X_train[:,1] = labelencoder_X_1.fit_transform(X_train[:,1])
labelencoder_X_6 = LabelEncoder()
X_train[:,6] = labelencoder_X_6.fit_transform(X_train[:,6])
# The Embarked columns has more than 2 levels, thus we need to use OneHotEncoder to convert it to dummy variables for each level
ct = ColumnTransformer([("Embarked", OneHotEncoder(), [6])], remainder = 'passthrough')
X_train = ct.fit_transform(X_train)
X_train = X_train[:, 1:]
# Load the test data using pandas and perform all the similar transformations to achieve consistency across training and test data.
test_data_df = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data_df.head()
test_data = test_data_df.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1)
test_data.info()
test_data[['Age']] = test_data[['Age']].fillna(value=test_data.Age.mean())
test_data[['Fare']] = test_data[['Fare']].fillna(value=test_data.Fare.mean())
X_test = test_data.iloc[:,0:8].values
labelencoder_X_1 = LabelEncoder()
X_test[:,1] = labelencoder_X_1.fit_transform(X_test[:,1])
labelencoder_X_6 = LabelEncoder()
X_test[:,6] = labelencoder_X_6.fit_transform(X_test[:,6])
ct = ColumnTransformer([("Embarked", OneHotEncoder(), [6])], remainder = 'passthrough')
X_test = ct.fit_transform(X_test)
X_test = X_test[:, 1:]
# Standardize the values in test and train to avoid problems due to different range of the variables
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Create a classifier using Sequential() funtion in Keras  
classifier = Sequential()
# Add 4 hidden layers in the neural network with 6 units each. Using relu activiation for hidden layer. 
classifier.add(Dense(units = 6, kernel_initializer  = 'random_uniform', activation = 'relu', input_dim = 8))
classifier.add(Dense(units = 12, kernel_initializer  = 'random_uniform', activation = 'relu'))
classifier.add(Dense(units = 12, kernel_initializer  = 'random_uniform', activation = 'relu'))
classifier.add(Dense(units = 12, kernel_initializer  = 'random_uniform', activation = 'relu'))
classifier.add(Dense(units = 8, kernel_initializer  = 'random_uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer  = 'random_uniform', activation = 'sigmoid'))
# Compile the model, we are using ADAM as the optimizer and binary_crossentropy for loss since we are doing a classifier problem.
classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy', metrics = ['accuracy'])
# Fit the model on the train data
classifier.fit(X_train, Y_train, batch_size=10, epochs = 100)
# Make predictions
y_pred = classifier.predict(X_test)
# Set cutoff as 0.5
y_pred = y_pred > 0.5
# Add predictions to output dataframe
output = pd.DataFrame({'PassengerId': test_data_df.PassengerId, 'Survived': y_pred.astype(int).flatten()})
# Export the submission file as CSV
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")