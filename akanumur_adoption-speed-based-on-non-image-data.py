#Required Basic Libraries

import numpy as np

import tensorflow as tf

import time

import pandas as pd



#Machine Learning

from sklearn.model_selection import train_test_split 



from sklearn.datasets import make_classification

from sklearn.preprocessing import OneHotEncoder, StandardScaler,MinMaxScaler,LabelEncoder

#We cannot use the OrdinalEncoder as we classes in numerical 



#for modelling

from tensorflow.keras.regularizers import l2

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Activation, LeakyReLU

from tensorflow.keras.utils import to_categorical

from keras import regularizers 
train_df = pd.read_csv("../input/train/train.csv")

test_df = pd.read_csv("../input/test/test.csv")
train_df.head(2)
train_df.describe()
train_df.dtypes
# define target variable and eliminate the less useful metrics

target = train_df['AdoptionSpeed']
features = train_df.drop(columns=['Name', 'RescuerID', 'Description', 'PetID', 'AdoptionSpeed'])

features_test = test_df.drop(columns=['Name', 'RescuerID', 'Description', 'PetID'])

features.head()
# standardize features

scaler = MinMaxScaler()

features = scaler.fit_transform(features)

features[ :1]

print(features)
# split features and targets into train/test sets

# random_state set at 14 or 21 for specific results



X_train, X_test, Y_train, Y_test = train_test_split(

        features, target, random_state=21) 
print(X_train)
X_train.shape
Y_train.head()
Y_train_1h = to_categorical(Y_train)

Y_train_1h[0:3, ]
Y_test.head()
# encode class values as integers

Y_test_1h = to_categorical(Y_test)

Y_test_1h[0:3,]
#first input layer should be the number of input features

#output layer should be number of classes

number_of_features = 19

model = Sequential()

model.add(Dense(19, activation='relu',kernel_regularizer=regularizers.l2(0.0001)))

model.add(Dense(250, activation='relu',kernel_regularizer=regularizers.l2(0.0001)))

model.add(Dense(100, activation='relu',kernel_regularizer=regularizers.l2(0.0001)))

model.add(Dense(75, activation='relu',kernel_regularizer=regularizers.l2(0.0001)))

model.add(Dense(50, activation='relu',kernel_regularizer=regularizers.l2(0.0001)))

model.add(Dense(5, activation='softmax'))
#Compile neural network

model.compile(loss='categorical_crossentropy',

              optimizer="adam",

              metrics=['accuracy']

             )
#Train Neural net

stime = time.monotonic()

history = model.fit(X_train, Y_train_1h, epochs=10, batch_size=19)

etime = time.monotonic()



print('training time (s): ', etime-stime)
print(Y_test_1h)
print(model.evaluate(X_test, Y_test_1h))
# standardize test features

scaler = MinMaxScaler()

test_features = scaler.fit_transform(features_test)

test_features[0]
predicted_target = model.predict(test_features)

predicted_target[0]
test_df.index
print(predicted_target)
y_pred = pd.DataFrame(model.predict_classes(test_features), index = test_df.index)

y_pred[0].unique()
y_pred['PetID'] = test_df['PetID']
y_pred['PetID'] = test_df['PetID']

y_pred.rename(columns={0:'AdoptionSpeed'}, inplace=True)

y_pred.head(2)
y_pred = y_pred[['PetID','AdoptionSpeed']]

y_pred.head(2)
y_pred.to_csv('submission.csv', index=False)