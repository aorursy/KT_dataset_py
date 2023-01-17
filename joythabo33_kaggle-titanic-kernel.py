#Imports
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from keras.layers import Dense
from keras import Input
from keras import Model
from keras.optimizers import Adam, SGD
import os
#Data set paths
train_path = "../input/train.csv"
test_path = "../input/test.csv"
#Reading the data
df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)
df_train.head()
df_test.head()
df_train.describe()
df_test.describe()
df_train[["Age"]]
df_test[["Age"]]
#Our Important Features(For now) are SibSp, Class, Sex, Survived(target output) and Age
train_data = df_train[["SibSp", "Parch", "Pclass", "Sex", "Age", "Survived"]].replace("male", 1).replace("female", 0)
test_data = df_test[["SibSp", "Parch","Pclass", "Sex", "Age"]].replace("male", 1).replace("female", 0)
train_data.fillna(train_data.mean(),inplace=True)
test_data.fillna(test_data.mean(),inplace=True)
#The .fillna method will impute mean values in cells that have no data
#The .replace method will change male strings to 1 and female strings to 0
x_train = train_data[["SibSp","Parch", "Pclass", "Sex", "Age"]]
x_test = test_data[["SibSp", "Parch", "Pclass", "Sex", "Age"]]
y_train = train_data[["Survived"]]
n_inputs = len(x_train.columns) #Number of inputs(columns/nodes/neurons)
n_outputs = len(y_train.columns)#Number of outputs(columns/nodes/neurons)
inputs = Input(shape = (n_inputs, ))
hl = Dense(units=10, activation="relu")(inputs) #Hidden Layer 1
hl = Dense(units=10, activation="relu")(hl) #Hidden Layer 2
outputs = Dense(units=1, activation="sigmoid")(hl)
model = Model(inputs, outputs)
model.summary()
optimizer = Adam(lr=0.0001,decay=1e-6) #Learning rate decay
pi = 3.14159265359
golden_ratio = 1.61803398875
model.compile(loss="binary_crossentropy", metrics=["accuracy"], optimizer=optimizer)
history =  model.fit(x_train, y_train, epochs=int(len(x_train)*pi),validation_split = 0.2, batch_size=int(len(x_train)/golden_ratio), verbose=2)
prediction = model.predict(x_test, batch_size = int(len(x_train)/golden_ratio))


model.save("Kaggle Titanic Comp.h5")
'''
Source: Jason Brownlee
Site: https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
'''

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()
'''
Source: Jason Brownlee
Site: https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
'''
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()
