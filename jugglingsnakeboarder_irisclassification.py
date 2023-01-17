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
import pandas as pd

iris_df = pd.read_csv("../input/iris/Iris.csv")

iris_df
# there are no entries with NaN

iris_df.isna().sum()
from sklearn.preprocessing import MinMaxScaler

minmax_scaler = MinMaxScaler(feature_range=(0, 1))

iris_df['SepalLengthCm']= minmax_scaler.fit_transform(iris_df[['SepalLengthCm']])

iris_df['SepalWidthCm']= minmax_scaler.fit_transform(iris_df[['SepalWidthCm']])

iris_df['PetalLengthCm']= minmax_scaler.fit_transform(iris_df[['PetalLengthCm']])

iris_df['PetalWidthCm']= minmax_scaler.fit_transform(iris_df[['PetalWidthCm']])

iris_df
iris_input = iris_df

iris_output = iris_df['Species']

iris_input = iris_input.drop(['Species'], axis=1)

iris_input = iris_input.drop(['Id'], axis=1)

print(iris_input)

print(iris_output)

remember_iris_output=iris_output

iris_output.shape

from sklearn import preprocessing

my_label_encoder = preprocessing.LabelEncoder()

my_label_encoder.fit(iris_output)

iris_output = my_label_encoder.transform(iris_output)

from keras.utils import to_categorical

iris_output = to_categorical(iris_output)

print(iris_output)
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(iris_input, iris_output, test_size=0.2, random_state=1)

X_train, X_val, Y_train, Y_val   = train_test_split(X_train, Y_train, test_size=0.1, random_state=1)

print("X_train.shape: ",X_train.shape)

print("Y_train.shape: ", Y_train.shape)

print("X_test.shape: ",X_test.shape)

print("Y_test.shape: ",Y_test.shape)

print("X_val.shape: ",X_val.shape)

print("Y_val.shape: ",Y_val.shape)

print(Y_test)
# my  MLP with regularization 

from keras import regularizers

from keras.models import Sequential

from keras.layers.core import Dense # MLP 



model = Sequential()

model.add(Dense(10, kernel_regularizer=regularizers.l2(0.001),activity_regularizer=regularizers.l2(0.001),activation='tanh', input_shape = (4,)))

model.add(Dense(8,kernel_regularizer=regularizers.l2(0.001),activity_regularizer=regularizers.l2(0.001), activation='tanh'))

model.add(Dense(6,kernel_regularizer=regularizers.l2(0.001),activity_regularizer=regularizers.l2(0.001), activation='tanh'))

model.add(Dense(3, activation='softmax'))# !



# optimizer 'adam' produces the best results

model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['acc'])





#Keras needs a numpy array as input and not a pandas dataframe

print(X_train)

print(Y_train)



history = model.fit(X_train, Y_train,

                    shuffle=True,

                    batch_size=64,

                    epochs=1000,

                    verbose=2,

                    validation_data=(X_val, Y_val))

import matplotlib.pyplot as plt

# Plot training & validation accuracy values

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()
# have a look at my results

eval_train = model.evaluate(X_train,Y_train)

print(eval_train)

eval_val = model.evaluate(X_val,Y_val)

print(eval_val)

eval_test = model.evaluate(X_test,Y_test)

print(eval_test)
# my prediction 

print(X_test)

results = model.predict(X_test)

results = (results > 0.5).astype(int)

results

#results.shape
text_pred = list(my_label_encoder.inverse_transform(results.argmax(1)))

print(text_pred)

print(len(text_pred))
Y_pred = results

print(Y_test)

print(Y_pred)
#Accuracy of the predicted values

from sklearn.metrics import classification_report

iris_names = ['1-0-0 = iris setosa', '0-1-0 = iris versicolor', '0-0-1 = iris virginica']

print(classification_report(Y_test,Y_pred,target_names=iris_names))
