#Importing all the libraries we need



import os

import random

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_validate

from keras.layers import Dense , Dropout

from keras.models import Sequential



for dir in os.walk("/kaggle/input/"):

    print(dir[2])
# Getting the training data



training_data = pd.read_csv('../input/train.csv')



# Printing first fice instances of training data

training_data.head(10)
# Printing last five instances of training data



training_data.tail(10)
# Getting a total number of data values we have



print(len(training_data))
# Checking for null values



training_data.isna().sum()
# Droping the columns that are not necessary



training_data = training_data.drop(columns = ['Ticket','Name','Cabin','PassengerId'])
training_data.isna().sum()
# Replacing Null values in dataset with mean values



mean_value = training_data['Age'].mean()

training_data['Age'] = training_data['Age'].fillna(mean_value) 
training_data.isna().sum()
# Dropping the values that are null



training_data = training_data.dropna()
print(len(training_data))
training_data.head()
training_data.info()
training_data.describe()
# This will give us the count of unique values present in Survived column



training_data['Survived'].value_counts()
# Plotting a graph for visualization



training_data['Survived'].value_counts().plot.bar()
#Generating Testing data



testing_data = pd.read_csv("../input/test.csv")
# First 10 instances of testing_data 



testing_data.head(10)
# Last 10 instances of testing_data



testing_data.tail(10)
# Getting the total number of instances in testing_data



print(len(testing_data))
# Getting count of Na values



testing_data.isna().sum()
# Droping columns that are not necessary

passenger_id = pd.DataFrame(testing_data['PassengerId'])

testing_data = testing_data.drop(columns = [ 'PassengerId' , 'Cabin' , 'Name' , 'Ticket'])

passenger_id.head()

print(len(passenger_id))
# Filling the null values with mean values



mean_value = testing_data['Age'].mean()

testing_data['Age'] = testing_data['Age'].fillna(mean_value)

mean_value = testing_data['Fare'].mean()

testing_data['Fare'] = testing_data['Fare'].fillna(mean_value)
testing_data = testing_data.dropna()
testing_data.isna().sum()
print(len(testing_data))
# Reading the actual labels for test data



gender_submission = pd.read_csv("../input/gender_submission.csv")

gender_submission.head()
len(gender_submission)
gender_submission['Survived'].value_counts().plot.bar()
training_data.head()
# Encoding the values from column Sex and Embarked



enc = LabelEncoder()

training_data['Sex'] = enc.fit_transform(training_data['Sex'])

training_data['Embarked'] = enc.fit_transform(training_data['Embarked'])
training_data.head()
training_data['Sex'].value_counts().plot.bar()
training_data['Embarked'].value_counts().plot.bar()
sns.pairplot(training_data,hue="Survived")
# Generating trianing data



X_train = training_data.iloc[:,1:]

Y_train = np.array(training_data['Survived'])
# Converting it into numpy array



X_train = np.array(X_train)

print(X_train.shape)
Y_train = np.array(Y_train)

print(Y_train.shape)
print(X_train[0,:])
print(X_train[0:5])
print(Y_train[0:5])
# Splitting training data into train and test, becuase we don't have test data here and the test data in test.csv is for prediction purpose so we will work on training data



X_t , x_test , Y_t , y_test = train_test_split(X_train,Y_train)
Y_t.shape
# Creating our Neural Network



model = Sequential()



# First Hidden layer with 256 neurons

model.add(Dense(256 , activation = 'sigmoid' , input_dim = (7)))



# Second Hideen layer with 256 neurons

model.add(Dense(256 , activation = 'relu'))



# Third Hidden layer with 128 neurons

model.add(Dense(128 , activation = 'sigmoid'))



# Fourth Hidden layer with 128 neurons

model.add(Dense(128 , activation = 'relu'))



model.add(Dropout(0.5))



model.add(Dense(1 , activation = 'sigmoid'))
# Defining rules for our Neural Netowrk



model.compile(loss = 'binary_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'])
# Fitting data to our model



model.fit( X_t , Y_t , epochs=50 , batch_size=32)
# Evaluating our model on test data



model.evaluate(x_test,y_test , batch_size = 32)
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()

classifier.fit(X_t,Y_t)
# Evaluating on test data

classifier.score(x_test,y_test)
from sklearn.ensemble import RandomForestClassifier

classifier_2 = RandomForestClassifier()

classifier_2.fit(X_t,Y_t)
# Evaluating on test data

classifier_2.score(x_test,y_test)
from sklearn.ensemble import GradientBoostingClassifier

classifier_3 = GradientBoostingClassifier()

classifier_3.fit(X_t,Y_t)
# Evaluating on test data

classifier_3.score(x_test,y_test)
# Cross validation on Logistic Regression

result = cross_validate(classifier , X_train , Y_train , cv=5)

print(result)
# Cross validation on Random Forest Classifier

result = cross_validate(classifier_2 , X_train , Y_train , cv=5)

print(result)
# Cross validation on Gradient Boosting Classifier

result = cross_validate(classifier_3 , X_train , Y_train , cv=5)

print(result)
print(type(testing_data))

print(len(testing_data))

print(testing_data[0:5])
# Encoding 'Sex' and 'Embarked' column of testing_data

testing_data['Sex'] = enc.fit_transform(testing_data['Sex'])

testing_data['Embarked'] = enc.fit_transform(testing_data['Embarked'])
# Forst five instances of testing_data

testing_data.head()
# X_pred is variable that stores values to be predicted

X_pred = np.array(testing_data)
print(X_pred[0:5])
X_pred.shape
# Predicting values, here Y_pred contains predicted values 

Y_pred = model.predict(X_pred).round()
# Y_test contains the actual labels for our prediction data

Y_test = np.array(gender_submission)

Y_test = Y_test[:,1]
print(Y_test)

print(Y_test.shape)
Y_pred = Y_pred.reshape(418,)

print(Y_pred)

print(Y_pred.shape)
cm = confusion_matrix(Y_test , Y_pred)



plt.subplots(figsize = (10,8))



sns.heatmap(cm , xticklabels = ['Survived' , 'Dead'] , yticklabels = ['Survived','Dead'])
Y_pred = classifier.predict(X_pred).round()

print(Y_pred)
cm = confusion_matrix(Y_test , Y_pred)



plt.subplots(figsize = (10,8))



sns.heatmap(cm , xticklabels = ['Survived' , 'Dead'] , yticklabels = ['Survived','Dead'])
Y_pred = classifier_2.predict(X_pred).round()

print(Y_pred)

#Y_pred = pd.DataFrame(Y_pred , columns=["Survived"])

#passenger_id['Survived'] = Y_pred

#print(len(passenger_id))

#print(len(Y_pred))

#passenger_id['Survived'] = Y_pred

#passenger_id.head(10)

#passenger_id.head(10)

#Y_pred.head(10)

#predictions = pd.concat([passenger_id,Y_pred] , axis=1, join='inner')

#predictions.head()
cm = confusion_matrix(Y_test , Y_pred)



plt.subplots(figsize = (10,8))



sns.heatmap(cm , xticklabels = ['Survived' , 'Dead'] , yticklabels = ['Survived','Dead'])
Y_pred = classifier_3.predict(X_pred).round()

print(Y_pred)

print(len(passenger_id))

print(len(Y_pred))

passenger_id['Survived'] = Y_pred

passenger_id.head(10)

passenger_id.to_csv("predictions.csv" , index=False)
cm = confusion_matrix(Y_test , Y_pred)



plt.subplots(figsize = (10,8))



sns.heatmap(cm , xticklabels = ['Survived' , 'Dead'] , yticklabels = ['Survived','Dead'])