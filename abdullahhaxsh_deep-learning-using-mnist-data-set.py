import numpy as np 
import pandas as pd 
import tensorflow
import keras
from keras import datasets
# ##################### LOADING DATA SETS ############### #
data = datasets.mnist
(x_train , y_train) , (x_test , y_test) = data.load_data()
x_train , x_test = x_train/255.0 , x_test/255.0 ##### Data normalization to quicken calculations #####
print(x_train.shape , x_test.shape)

# ################### Neural network ################### #

model = keras.Sequential([
keras.layers.Flatten( input_shape = (28,28) ),
keras.layers.Dense( 128 , activation='relu'),
keras.layers.Dense( 10 , activation='softmax' )

])
# model.add(keras.layers.Flatten( input_shape = (28,28) ))
# model.add(keras.layers.Dense(128 , activation='relu'))
# model.add(keras.layers.Dense(10 , activation='softmax'))

model.compile(optimizer='adam' , loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'] )

model.fit(x_train , y_train , epochs = 5)

loss , acc = model.evaluate(x_test , y_test)

predictions = model.predict(x_test)


for i in range(len(predictions)):
  print('Pre : '+str(np.argmax(predictions[i])), '\t' , 'Act : '+str(y_test[i]))
  if np.argmax(predictions[i]) != y_test[i]:
    print('\n'+' ### ERROR ### '+'\n')
print('Accuracy : ' , acc*100)
print(' ### MODEL TRAINED ### ')    