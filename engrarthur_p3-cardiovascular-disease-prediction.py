import sys
import pandas as pd
import numpy as np
import sklearn
import matplotlib
import keras


import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
cardio = pd.read_csv('../input/cardio1/heart.csv')
print( 'Shape of DataFrame: {}'.format(cardio.shape))
print (cardio.loc[1])
cardio.loc[280:]
data = cardio[~cardio.isin(['?'])]
data.loc[280:]
data = data.dropna(axis=0)
data.loc[280:].astype(float)
print(data.shape)
print(data.dtypes)
data = data.apply(pd.to_numeric)
data.dtypes
data.describe()
data.hist(figsize = (12, 12))
plt.show()
from sklearn import model_selection

X = np.array(data.drop(['LDL_Cholesterol'], 1))
y = np.array(data['LDL_Cholesterol'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2)
from keras.utils.np_utils import to_categorical

y_train = to_categorical(y_train, num_classes=None)
y_test = to_categorical(y_test, num_classes=None)
print (X_train.shape, y_train.shape)
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout


def create_model():
    # create model
    model = Sequential()
    layer = Dropout(0.5)
    model.add(Dense(32, input_dim=8, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5, input_shape=(2,)))
    model.add(Dense(32, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5, input_shape=(2,)))
    model.add(Dense(126, activation='softmax'))
    
    # compile model
    adam = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model

model = create_model()

print(model.summary())
history=model.fit(X_train, y_train, validation_data=(X_test, y_test),epochs=200, batch_size=10, verbose = 10)

import matplotlib.pyplot as plt
%matplotlib inline
# Model accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()
y_pred = model.predict(X_test)
y_pred[:5]


y_test[:5]


score = model.evaluate(X_test, y_test,verbose=1)

print('Test score:', score)
from matplotlib import pyplot

model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae', 'mape', 'cosine', 'accuracy'])
history = model.fit(X_train, y_train, epochs=100)
cardio.loc[(cardio['Gender'] >= 1) & (cardio['Age'] >= 18) & (cardio['Age'] <= 44) & (cardio['LDL_Cholesterol'] >= 60) & (cardio['LDL_Cholesterol'] <= 130) & (cardio['BMI'] >= 18.5) & (cardio['BMI'] <= 24.9) & (cardio['WaistCircum'] <= 102), 'CVD'] = "Low"
cardio.loc[(cardio['Gender'] >= 2) & (cardio['Age'] >= 18) & (cardio['Age'] <= 54) & (cardio['LDL_Cholesterol'] >= 60) & (cardio['LDL_Cholesterol'] <= 130) & (cardio['BMI'] >= 18.5) & (cardio['BMI'] <= 24.9) & (cardio['WaistCircum'] <= 88),'CVD'] = "Low"
cardio.loc[(cardio['Gender'] >= 1) & (cardio['Age'] >= 45) & (cardio['Age'] <= 200) & (cardio['LDL_Cholesterol'] >= 131) & (cardio['LDL_Cholesterol'] <= 500) & (cardio['BMI'] >= 25) & (cardio['BMI'] <= 39.9) & (cardio['WaistCircum'] >= 102), 'CVD'] = "High"
cardio.loc[(cardio['Gender'] >= 2) & (cardio['Age'] >= 55) & (cardio['Age'] <= 200) & (cardio['LDL_Cholesterol'] >= 131) & (cardio['LDL_Cholesterol'] <= 500) & (cardio['BMI'] >= 25) & (cardio['BMI'] <= 39.9) & (cardio['WaistCircum'] >= 88), 'CVD'] = "High"





cardio.head(10)
cardio.dropna()
by_cat_gen = cardio.groupby(['Gender','CVD'])

by_cat_gen.size().plot(kind='bar')