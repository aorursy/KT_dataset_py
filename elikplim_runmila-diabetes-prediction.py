import pandas

from keras.models import Sequential

from keras.layers import Dense

#from keras.layers import Dropout

import numpy

# fix random seed for reproducibility

numpy.random.seed(7)



#names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

data = pandas.read_csv('../input/diabetes.csv')



array = data.values

X = array[:,0:8]

Y = array[:,8]



from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1

                                                    ,random_state = 0)



data.head(5)
model = Sequential()

model.add(Dense(15, input_dim=8, kernel_initializer='uniform', activation='relu'))

#model.add(Dropout(0.2))

model.add(Dense(8, kernel_initializer='uniform', activation='relu'))

#model.add(Dropout(0.2))

model.add(Dense(3, kernel_initializer='uniform', activation='relu'))

model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))



model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
from tensorflow.keras.callbacks import TensorBoard

import datetime



log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = TensorBoard("logs")



model.fit(X_train, Y_train, 

          epochs=200, 

          batch_size=10, 

          validation_data=(X_test, Y_test), 

          callbacks=[tensorboard_callback])
%load_ext tensorboard.notebook

%tensorboard --logdir logs


scores = model.evaluate(X_test, Y_test)

print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
from keras.models import model_from_json



# Save model Architecture as JSON

model_json = model.to_json()

with open(r'diabetes_model.json', "w") as json_file:

   json_file.write(model_json)



# Save weights as HDF5

model.save_weights(r'diabetes_model.h5')