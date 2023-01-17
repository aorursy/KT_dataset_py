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
# load the train data

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()
# load the test data

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
# setting passangerId as index column

train_data.set_index('PassengerId',inplace = True)

test_data.set_index('PassengerId',inplace = True)

print(train_data.head())

print(test_data.head())
# droping Name column from test and train 

train_data.drop(['Name','Ticket','Cabin'],axis = 1, inplace = True)

test_data.drop(['Name','Ticket','Cabin'],axis = 1, inplace = True)

print(train_data.head())

print(test_data.head())
# creating Training data and lables

train_y = train_data["Survived"]

print(train_y.head())

train_X = train_data.drop("Survived",axis = 1)

print(train_X.head())

# Missing values in train data:in age column

age_null_count = train_X['Age'].isna().sum()

print(age_null_count)



# missing value imputation : for Missing value we will generate random numbers by taking mean and standard deviation



avg_age = train_X['Age'].mean()

std_age = train_X['Age'].std()

age_temp  = np.random.randint(avg_age - std_age,avg_age + std_age, size = age_null_count)

print(age_temp)

age_slice = train_X["Age"].copy()

age_slice[np.isnan(age_slice)] = age_temp

train_X["Age"] = age_slice
# Missing values:in embarked column we can use mode because it takes most frequest value

embarked_count = train_X['Embarked'].isna().sum()

print(embarked_count)

embarked_temp = train_X['Embarked'].mode()

train_X['Embarked'] = train_X['Embarked'].fillna(embarked_temp)
test_age = test_data['Age'].isna().sum()

print(test_age)

avg_age = test_data['Age'].mean()

std_age = test_data['Age'].std()

age_temp  = np.random.randint(avg_age - std_age,avg_age + std_age, size = test_age)

print(age_temp)

age_slice = test_data["Age"].copy()

age_slice[np.isnan(age_slice)] = age_temp

test_data["Age"] = age_slice

# Missing values in train data:in age column

Fare_null_count =  test_data['Fare'].isna().sum()

print(Fare_null_count)



# missing value imputation : for Missing value we will generate random numbers by taking mean and standard deviation



avg_fare = test_data['Fare'].mean()

std_fare = test_data['Fare'].std()

fare_temp  = np.random.randint(avg_fare - std_fare,avg_fare + std_fare, size = Fare_null_count)

print(age_temp)

fare_slice = test_data['Fare'].copy()

fare_slice[np.isnan(fare_slice)] = fare_temp

test_data['Fare'] = fare_slice
# Scaling numerical features:using Standard scalar

train_data_numerical_features = list(train_X.select_dtypes(include = ['int64', 'float64', 'int32']).columns)

print(train_data_numerical_features)





# we will use standard scalar

from sklearn.preprocessing import StandardScaler

ss_scaler = StandardScaler()

train_data_ss = pd.DataFrame(data = train_X)

train_data_ss[train_data_numerical_features] = ss_scaler.fit_transform(train_data_ss[train_data_numerical_features])

print(train_data_ss)



    
# Scaling numerical features-test data:using Standard scalar

test_data_numerical_features = list(test_data.select_dtypes(include = ['int64', 'float64', 'int32']).columns)

print(test_data_numerical_features)





# we will use standard scalar

from sklearn.preprocessing import StandardScaler

ss_scaler = StandardScaler()

test_data_ss = pd.DataFrame(data = test_data)

test_data_ss[test_data_numerical_features] = ss_scaler.fit_transform(test_data_ss[test_data_numerical_features])

print(test_data_ss)
# Scaling categorical features:One hot encoding

train_data_categorical_features = list(train_X.select_dtypes(include = ['object']).columns)

print(train_data_categorical_features)



for i in train_data_categorical_features:

    train_data_ss = pd.concat([train_data_ss,pd.get_dummies(train_data_ss[i], prefix=i)],axis=1)

    train_data_ss.drop(i, axis = 1, inplace=True)



# print(train_data_ss)

print(train_data_ss.head())
# Scaling categorical features - test data:One hot encoding

test_data_categorical_features = list(test_data.select_dtypes(include = ['object']).columns)

print(test_data_categorical_features)



for i in test_data_categorical_features:

    test_data_ss = pd.concat([test_data_ss,pd.get_dummies(test_data_ss[i], prefix=i)],axis=1)

    test_data_ss.drop(i, axis = 1, inplace=True)



# print(train_data_ss)

print(test_data_ss.head())

print(test_data_ss.isnull().sum())
# implementing Randomforest classifier

from sklearn.ensemble import RandomForestClassifier



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)



model.fit(train_data_ss,train_y)



predictions = model.predict(test_data_ss)

print(test_data_ss.index)

print(predictions)
output = pd.DataFrame({'PassengerId':test_data.index, 'Survived': predictions})

output.to_csv('my_submission2.csv', index=False)

print("Your submission was successfully saved!")
# implementing logistic regression

from sklearn.linear_model import LogisticRegression



clf = LogisticRegression()



clf.fit(train_data_ss,train_y)
lr_predictions = clf.predict(test_data_ss)

print(test_data_ss.index)

print(lr_predictions)
output = pd.DataFrame({'PassengerId':test_data.index, 'Survived': lr_predictions})

output.to_csv('logisticRegression.csv', index=False)

print("Your submission was successfully saved!")
import keras

num_classes = 2

print(train_data_ss.shape)

print(train_y.shape)

y_train = keras.utils.to_categorical(train_y, num_classes)
# importing all dependencies

from keras.models import Sequential

from keras.layers import Dense
# Network creation :

model = Sequential()

model.add(Dense(32,input_shape=(10,),activation = 'relu'))

model.add(Dense(64,activation = 'relu'))

model.add(Dense(128,activation = 'relu'))

model.add(Dense(2,activation = 'softmax'))
# calculating loss and optimization

model.compile(loss = 'binary_crossentropy',

             optimizer = 'rmsprop',

             metrics =['accuracy'])
model.fit(train_data_ss,y_train, epochs = 1500, batch_size = 128 )
pred = model.predict(test_data_ss)

result = [np.argmax(x) for x in pred]

print(result)

output = pd.DataFrame({'PassengerId':test_data.index, 'Survived': result})

output.to_csv('/kaggle/working/NeuralNetwork2.csv', index=False)

print("Your submission was successfully saved!")