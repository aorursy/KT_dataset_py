import pandas as pd

import numpy as np

from subprocess import check_output

import keras

from keras.models import Sequential

from keras.layers import Dense,Activation,Conv2D,MaxPool2D,Flatten,Dropout

from keras.utils import np_utils

from sklearn.cross_validation import train_test_split



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
df_train = pd.read_csv("../input/train.csv")

df_train.head(3)
df_test = pd.read_csv("../input/test.csv")

df_test.head(3)
#List of features

features = ["%s%s" %("pixel",pixel_no) for pixel_no in range(0,784)]

df_train_features = df_train[features]

df_train_features.shape
#Test Data shape

df_test.shape
#Convert single digit to one dimentional array

df_train_labels = df_train["label"]

df_train_labels_categorical = np_utils.to_categorical(df_train_labels)

df_train_labels_categorical[0:3]
#Train test split to train and test model

X_train,X_test,y_train,y_test = train_test_split(df_train_features,df_train_labels_categorical,test_size=0.10,random_state=32)
#Check final shape of dataset

print("X_train shape ",X_train.shape)

print("y_train shape ",y_train.shape)

print("X_test shape ",X_test.shape)

print("y_test shape ",y_test.shape)
#Architecture of keras model

model = Sequential()

model.add(Dense(32,activation='relu',input_dim=784))

model.add(Dense(64,activation='relu'))

model.add(Dropout(0.03))

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.03))

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.25))

model.add(Dense(10, activation='softmax'))



model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer=keras.optimizers.Adadelta(),

              metrics=['accuracy'])



model.fit(X_train.values, y_train,

          batch_size=128,

          epochs=50,

          verbose=1)
#Predict the digit for given input

pred_classes = model.predict_classes(df_test.values)
#Finally generate kaggle submission file

submission = pd.DataFrame({

    "ImageId": df_test.index+1,

    "Label": pred_classes})

print(submission[0:10])



submission.to_csv('./keras_model_1.csv', index=False)