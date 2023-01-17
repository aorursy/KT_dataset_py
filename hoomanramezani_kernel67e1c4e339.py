# Import libraries
# Data analysis
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random as rnd

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
train_df = pd.read_csv('/kaggle/input/titanic/train.csv')
test_df = pd.read_csv('/kaggle/input/titanic/test.csv')
combine = [train_df, test_df]
combine[1].head()

y_train = train_df['Survived']
# Analyze the data
train_df.describe()
# Begin data wrangling

# Correct data: drop ticket, passengerId and cabin due to incomplete data and unlikely correlation to survival
train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]
# Test if title can predict if someone survived, get all title names
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    
pd.crosstab(train_df['Title'], train_df['Sex'])
# Add common titles to a set and classify the rest as rare
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
    'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
# Title seems to have predictive power and will be converted to ordinal and added to the dataset
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

# Name can now be dropped
train_df = train_df.drop(['Name'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]

# Combine siblings and parents feature into one to create 'FamilySize', examine it's relationship to surviving
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean()
# No clear relation is found in familySize
# A better solution seems to be to combine all family sizes greater than one into a feature 'alone'
for dataset in combine:
    dataset['Alone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'Alone'] = 1
    
# Drop all other columns
train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize', 'PassengerId'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize', 'PassengerId'], axis=1)
combine = [train_df, test_df]
# Embarked has two missing fields, choose the most common to fill in the values with
frequent_port = train_df.Embarked.dropna().mode()[0]
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(frequent_port)
    
# Ordinalize the embarked category and add to the dataset   
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
# Fare has one missing value in test dataset, fill with median value
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

combine = [train_df, test_df]
# Map female to a 1 and male to 0
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
# Age has many missing columnns, in our investigation we noticed a correlation between age, gender and pclass
# Create an empty array size 2(Sex) x 3(Pclass) that contains guessed age values
guess_ages = np.zeros((2,3))

for dataset in combine:
    # Iterate over sex and pclass to guess age of each column
    for i in range(0, 2):
        for j in range(0, 3):
            # Drop null columns and aquire a guess based on columns specific sex and class
            guess_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j+1)]['Age'].dropna()
            age_guess = guess_df.median()
            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    # Fill in dropped columns with guesses
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

    
# FINISHED DATA WRANGLING
train_df = train_df.drop(['Survived'], axis=1)
train_df.head()

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D,MaxPool2D
import tensorflow as tf
import keras
# Scale values to help neural network training
continuous = ['Age', 'Fare', 'Pclass']

scaler = StandardScaler()

for var in continuous:
    train_df[var] = train_df[var].astype('float64')
    train_df[var] = scaler.fit_transform(train_df[var].values.reshape(-1, 1))
for var in continuous:
    test_df[var] = test_df[var].astype('float64')
    test_df[var] = scaler.fit_transform(test_df[var].values.reshape(-1, 1))
    
train_df.head()
test_df.head()
# Stochastic Gratient Descent
# Adam algorithm for G.D.
# Use a sequential model, 1:1 input and output tensor in each layer
tf.keras.optimizers.Adam(
    learning_rate=0.00002, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
    name='Adam', 
)
# Creating a model with sequential columns
model = Sequential()
model.add(Flatten())

# Creates the first latyer with the input dimanetion. 
model.add(Dense(32 , input_dim=train_df.shape[1],kernel_initializer = 'uniform', activation='relu'))
model.add(Dense(32, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(32,kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(32,kernel_initializer = 'uniform', activation = 'relu'))

# Create output layer
# Feel free to experiment with the activation functions and the optimizers
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(train_df, y_train, epochs=250, batch_size=60, validation_split = 0.2)
print(model.summary())
scores = model.evaluate(train_df, y_train, batch_size=60)
y_pred = model.predict(test_df)

y_final = (y_pred > 0.5).astype(int).reshape(test_df.shape[0])
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

output = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': y_final})
output.to_csv('prediction.csv', index=False)
loss_train = history.history['loss']
loss_val = history.history['val_loss']
epochs = range(1,251)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
loss_train = history.history['accuracy']
loss_val = history.history['val_accuracy']
epochs = range(1,251)
plt.plot(epochs, loss_train, 'g', label='Training accuracy')
plt.plot(epochs, loss_val, 'b', label='validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()