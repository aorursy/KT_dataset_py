# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# Linear algebra
import numpy as np
# Data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd
from pandas.plotting import scatter_matrix
# Import keras deep learning library
import keras
from keras.models import Sequential
from keras.layers import Dense,MaxPooling2D,Flatten,Dropout
from keras.layers.convolutional import Conv2D
from keras import backend
# Globals constants
input_neurons=10
output_neurons=1
# Fix random seed for reproducibility
np.random.seed(7)
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
# Inspect the dataset, as you can see some columns have an empty value
train_df.info()
# See the head of the train dataset
train_df.head()
# For see the data's correlation between the different columns in the train dataset
_=scatter_matrix(train_df.drop('PassengerId', axis=1), figsize=(10, 10))
def simplify_ages(df):
    df['Age'] = df['Age'].fillna(df.Age.mean())
    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)
    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    categories = pd.cut(df['Age'], bins, labels=group_names)
    df['Age'] = categories.cat.codes 
    return df

def simplify_cabins(df):
    df['Cabin'] = df['Cabin'].fillna('N')
    df['Cabin'] = df['Cabin'].apply(lambda x: x[0])
    df['Cabin'] =  pd.Categorical(df['Cabin'])
    df['Cabin'] = df['Cabin'].cat.codes 
    return df

def simplify_fares(df):
    df['Fare'] = df.Fare.fillna(df.Fare.mean())
    bins = (-1, 0, 8, 15, 31, 1000)
    group_names = ['Unknown', 'First Quartile', 'Second Quartile',
                   'Third Quartile', 'Fourth Quartile']
    categories = pd.cut(df['Fare'], bins, labels=group_names)
    df['Fare'] = categories.cat.codes 
    return df

def simplify_sex(df):
    df['Sex'] = pd.Categorical(df['Sex'])
    df['Sex'] = df['Sex'].cat.codes 
    return df

def simplify_embarked(df):
    df['Embarked'] = df.Embarked.fillna(df.Embarked.mode()[0])
    df['Embarked'] = pd.Categorical(df['Embarked'])
    df['Embarked'] = df['Embarked'].cat.codes + 1
    return df

def normalize_titles(df):
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    df['Title'] = df['Title'].replace('Ms', 'Mrs')      
    df['Title'] = df['Title'].replace('Mrs', 'Mrs')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')     
    df['Title'] = df['Title'].replace('Miss', 'Miss')
    df['Title'] = df['Title'].replace('Master', 'Master')
    df['Title'] = df['Title'].replace('Mr', 'Mr')
    df['Title'] = df['Title'].replace('Capt', 'Officer')
    df['Title'] = df['Title'].replace('Major', 'Officer')
    df['Title'] = df['Title'].replace('Dr', 'Officer')
    df['Title'] = df['Title'].replace('Col', 'Officer')
    df['Title'] = df['Title'].replace('Rev', 'Officer') 
    df['Title'] = df['Title'].replace('Jonkheer', 'Royalty')    
    df['Title'] = df['Title'].replace('Don', 'Royalty')
    df['Title'] = df['Title'].replace('Dona', 'Royalty')
    df['Title'] = df['Title'].replace('Countess', 'Royalty')
    df['Title'] = df['Title'].replace('Lady', 'Royalty')
    df['Title'] = df['Title'].replace('Sir', 'Royalty')
    return df

def simplify_titles(df):
    df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    df=normalize_titles(df)
    df['Title'] = pd.Categorical(df['Title'])
    df['Title'] = df['Title'].cat.codes + 1
    return df

def simplify_family_size_and_is_alone(df):
    df['FamilySize'] = df ['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = 1
    df.loc[df['FamilySize'] > 1,'IsAlone'] = 0
    return df

def simplify_is_child(df):
    df['IsChild'] = 0
    df.loc[df['Age'] < 18,'IsChild'] = 1
    return df

def transform_features(df):
    df = simplify_titles(df)
    df= simplify_is_child(df)
    df = simplify_ages(df)
    df = simplify_cabins(df)
    df = simplify_family_size_and_is_alone(df)
    df = simplify_fares(df)
    df = simplify_sex(df)
    df = simplify_embarked(df)
    return df
train_df=transform_features(train_df)
test_df=transform_features(test_df)
# After of the simplify and normalize the data
train_df.info()
# Train Data Frame
xtrain_df = train_df.drop(['PassengerId','Ticket','Survived','Name','Parch','SibSp'], axis=1)
ytrain_df = train_df['Survived']
# Test Data Frame 
xtest_df = test_df.drop(['PassengerId','Ticket','Name','Parch','SibSp'], axis=1)
model = Sequential()
model.add(Dense(32, input_dim=input_neurons, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(5, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(output_neurons, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])
model.fit(xtrain_df, ytrain_df,epochs=50, batch_size=1,verbose=1)
scores = model.evaluate(xtrain_df, ytrain_df)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
predictions = model.predict_classes(xtest_df,verbose=0)
predictions=predictions.flatten()
results = pd.Series(predictions,name="Survived")
submission = pd.concat([pd.Series(range(892,1310),name = "PassengerId"),results],axis = 1)
submission.to_csv("titanic_datagen.csv",index=False)
# Clear error in tensorflow for session
backend.clear_session()