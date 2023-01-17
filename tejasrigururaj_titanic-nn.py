import math

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization

from tensorflow.keras import metrics

from tensorflow.keras import optimizers, regularizers

from tensorflow.keras.optimizers import Adam

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

print("Proceed.")

# Loading train and test data

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")



train_data.head()
# Converting the genders to 0 and 1: Female = 0, Male = 0



train_data.Sex = train_data.Sex.map({'female':0, 'male':1})

test_data.Sex = test_data.Sex.map({'female':0, 'male':1})

#train_data['Pclass'].isnull().values.any()

#train_data['Age'].isnull().sum()



# Age in train data is missing 177 entries



age_missing_train = []



for x in train_data['Age']:

    if math.isnan(x) == True:

        age_missing_train.append(1) # True

    else:

        age_missing_train.append(0) # False

        

train_data['Age_Missing'] = age_missing_train



mean_age_train = train_data['Age'].mean()

#print(mean_age)



train_data['Age'] = train_data['Age'].fillna(mean_age_train)
train_data.head()
#test_data['Age'].isnull().values.any()

#test_data['Age'].isnull().sum()



# Fare in test data is missing 1 value

# Age in test data is missing 86 values



age_missing_test = []



for x in test_data['Age']:

    if math.isnan(x) == True:

        age_missing_test.append(1) # True

    else:

        age_missing_test.append(0) # False

        

test_data['Age_Missing'] = age_missing_test



mean_age_test = test_data['Age'].mean()

mean_fare_test = test_data['Fare'].mean()



test_data['Age'] = test_data['Age'].fillna(mean_age_test)

test_data['Fare'] = test_data['Fare'].fillna(mean_fare_test)
y_train = train_data["Survived"] # Target for training data

features = ["Pclass","Sex","SibSp","Parch","Fare","Age", "Age_Missing"]

x_train = train_data[features] # Training data

x_test = test_data[features] # Validation data



x_train.head()

#print(len(x_train))



#print(X_test.ndim)
# Sequential network



model = Sequential()



model.add(Dense(8, activation='relu', input_dim=7))



model.add(Dense(8, activation='relu')) 



model.add(Dense(4, activation='relu'))

          

model.add(Dense(1, activation="sigmoid"))

          

model.compile(optimizer="adam", loss="binary_crossentropy", metrics="accuracy")



model.fit(x_train, y_train, batch_size=28, epochs=100)

model.summary()
predictions = model.predict(x_test)



for i in range(0,len(predictions)):

    if predictions[i] <= 0.5:

        predictions[i] = 0

    else:

        predictions[i] = 1

        

final_predictions = predictions.flatten()

        

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': final_predictions.astype(int)})

output.to_csv('my_submission.csv', index=False)



print(output.head())
