# initialization of the random generators for reproducible results

seed_value=0

from numpy.random import seed

seed(seed_value)

import tensorflow as tf

tf.random.set_seed(seed_value)
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
from keras.models import Sequential

from keras.layers.core import Dense # MLP 

from sklearn.preprocessing import LabelEncoder

import matplotlib as plt
# loading the train-data from train.csv-file with pandas 

train_df = pd.read_csv("/kaggle/input/titanic/train.csv")

train_df.head()
#loading the test-data from csv-file with pandas

test_df = pd.read_csv("/kaggle/input/titanic/test.csv")

test_df.head()

# the "Survived"-column is missing => to predict later
# have a look at the submission-file format

gender_sub_df = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")

gender_sub_df.head()

# this is the expected form of the csv-file to submit, containing only the passenger-ID und 0/1-"Survived"-Information

#gender_sub_df.shape #(418, 2)
# are there any NaN in the train-data?

train_df.isna().sum()
# are there any NaN in the test-data?

test_df.isna().sum()
# drop the features with too much NaN and 'ticket' with not trainable data from the train-dataframe

train=train_df.drop(['PassengerId','Name','Age','Ticket','Cabin'], axis=1)

train.head()

# later we must drop the "survived"-column for training
# drop the features with too much NaN and 'ticket' with not trainable data from the test-dataframe

test=test_df.drop(['PassengerId','Name','Age','Ticket','Cabin'], axis=1)

test.head()
# first candidate to be a key feature ist female/male survival ratio

survived_female = len(train[(train_df.Sex=='female') & (train.Survived==1)])/len(train[(train.Sex=='female')])

survived_male = len(train[(train.Sex=='male') & (train.Survived==1)])/len(train[(train.Sex=='male')])

print("survived females: ", survived_female)

print("survived males: ", survived_male)
# Passenger-Class 1,2,3 and survival seems to be a second key feature

survived_pclass1 = len(train_df[(train_df.Pclass==1) & (train_df.Survived==1)])/len(train_df[(train_df.Pclass==1)])

survived_pclass2 = len(train_df[(train_df.Pclass==2) & (train_df.Survived==1)])/len(train_df[(train_df.Pclass==2)])

survived_pclass3 = len(train_df[(train_df.Pclass==3) & (train_df.Survived==1)])/len(train_df[(train_df.Pclass==3)])

print("survived Pclass1:",survived_pclass1)

print("survived Pclass2:",survived_pclass2)

print("survived Pclass3:",survived_pclass3)
train.dtypes

# => sex and embarked are from object-datatype 
# two helpful functions to make the 'sex'- and 'embarked'-entrys from categorica to nummerical

def categorical2numerical_sex(value):

    if (value=='female'):

        return 1

    else:

        return 0

    

def categorical2numerical_embarked(value):

    if (value == "C"):

        return 2

    elif (value == "Q"):

        return 1

    else:

        return 0
# transform 'Sex' and 'Embarked' from categorical to numerical

train['Sex'] = train['Sex'].apply(categorical2numerical_sex)

train['Embarked'] = train['Embarked'].apply(categorical2numerical_embarked)

# the feature 'Fare' needs an [0..1] scaling for training in NN

from sklearn.preprocessing import MinMaxScaler

minmax_scaler = MinMaxScaler(feature_range=(0, 1))

train['Fare'] = minmax_scaler.fit_transform(train[['Fare']])

train
test.dtypes

# => sex and embarked are from object-datatype
# the same for test-data

test['Sex'] = test['Sex'].apply(categorical2numerical_sex)

test['Embarked'] = test['Embarked'].apply(categorical2numerical_embarked)

from sklearn.preprocessing import MinMaxScaler

minmax_scaler = MinMaxScaler(feature_range=(0, 1))

test['Fare'] = minmax_scaler.fit_transform(test[['Fare']])

test.shape # the shape fits to the gender_submission file
# drop the 'Survived' column to create the train-data

XTrain = train.drop(['Survived'], axis=1)

XTrain
# seperate the 'Survived'-colum to be the output of the NN

YTrain = train['Survived']

YTrain
# produce the train- and validate-data with sklearn "train_test_split"

from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(XTrain, YTrain, test_size=0.2, random_state=None )
# my 16-8-1-MLP with kerne- and activity-regularization at the inner layers => 79-80% acc

#from keras import regularizers

#model = Sequential()

#model.add(Dense(16, kernel_regularizer=regularizers.l2(0.001),activity_regularizer=regularizers.l2(0.001),activation='relu', input_shape = (6,)))

#model.add(Dense(8,kernel_regularizer=regularizers.l2(0.001),activity_regularizer=regularizers.l2(0.001), activation='relu'))

#model.add(Dense(1, activation='sigmoid'))
# my deep 64-52-32-20-16-1 MLP with regularization => 80-81% acc

from keras import regularizers

model = Sequential()

model.add(Dense(64, kernel_regularizer=regularizers.l2(0.001),activity_regularizer=regularizers.l2(0.001),activation='relu', input_shape = (6,)))

model.add(Dense(52,kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l2(0.01), activation='relu'))

model.add(Dense(32,kernel_regularizer=regularizers.l2(0.001),activity_regularizer=regularizers.l2(0.001), activation='relu'))

model.add(Dense(20,kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l2(0.001), activation='relu'))

model.add(Dense(16,kernel_regularizer=regularizers.l2(0.001),activity_regularizer=regularizers.l2(0.001), activation='relu'))

model.add(Dense(1, activation='sigmoid'))
# optimizer 'adam' produces the best results at my 16-8-1-MLP

model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['acc'])

#Keras needs a numpy array as input and not a pandas dataframe

X_train2 = X_train.values

#X_train2.reshape(-1,4948)

print(X_train2)

Y_train2 = Y_train.values

print(Y_train2)



history = model.fit(X_train2, Y_train2,

                    shuffle=True,

                    batch_size=16,

                    epochs=1000,

                    verbose=2,

                    validation_data=(X_val, Y_val))

# have a look at my results

eval_train = model.evaluate(X_train2,Y_train2)

print(eval_train)

eval_val = model.evaluate(X_val,Y_val)

print(eval_val)
# history-protocol data for plotting

history_dict = history.history

history_dict.keys()
# plot training & validation accuracy values

import matplotlib.pyplot as plt

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()
# plot training & validation loss values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'], loc='upper right')

plt.show()
# my prediction and submission

test2 = test.values

#test2.reshape(-1,2926)

#print(test2)

results = model.predict(test)

results = (results > 0.5).astype(int).reshape(test.shape[0])



submission = pd.DataFrame({'PassengerId': gender_sub_df['PassengerId'], 'Survived': results})

print(submission)

submission.to_csv('mysubmission.csv', index=False)