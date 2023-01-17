#loading python libs
import numpy as np
from pandas import read_csv
from pandas import DataFrame
from pandas import Series
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
# read csv with labels
dataframe = read_csv('../input/train.csv')
# remove NaNs from dataframe
def data_cleaning(dataframe):
    # Remove NaN
    dataframe['Age'].fillna(value=0, inplace=True)
    dataframe['Fare'].fillna(value=0.0, inplace=True)
    dataframe['Cabin'].fillna(value='UNKNOWN', inplace=True)
    dataframe['Embarked'].fillna(value='_', inplace=True)
    
    return dataframe

# convert categorical columns into numerical columns
def replace_categorical_to_numeric_data(dataframe):
    # Transform strings and categorical columns in to numerical columns
    dataframe['Sex'] = LabelEncoder().fit(dataframe['Sex']).transform(dataframe['Sex'])
    dataframe['Ticket'] = LabelEncoder().fit(dataframe['Ticket']).transform(dataframe['Ticket'])
    dataframe['Fare'] = LabelEncoder().fit(dataframe['Fare']).transform(dataframe['Fare'])
    dataframe['Cabin'] = LabelEncoder().fit(dataframe['Cabin']).transform(dataframe['Cabin'])
    dataframe['Embarked'] = LabelEncoder().fit(dataframe['Embarked']).transform(dataframe['Embarked'])
    
    return dataframe
# clean data
dataframe = data_cleaning(dataframe)
dataframe
# convert dataframe to be used to random forest
dataframe = replace_categorical_to_numeric_data(dataframe)
dataframe
# Choose label column (y) and feature columns (X)
y = dataframe.as_matrix(columns=['Survived'])
X = dataframe.as_matrix(['Pclass','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked'])
# Train random forest for the problem
model = RandomForestClassifier(verbose=1)
# Training
model = model.fit(X,np.ravel(y))
# Read test data
testdata = read_csv('../input/test.csv')
testdata = data_cleaning(testdata)
testdata = replace_categorical_to_numeric_data(testdata)
X_test = testdata.as_matrix(['Pclass','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked'])
y_pred = Series( model.predict(X_test) )
answer = DataFrame({ 'PassengerId': testdata['PassengerId'], 'Survived': y_pred })
answer