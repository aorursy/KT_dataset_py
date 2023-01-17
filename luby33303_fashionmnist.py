import numpy as np
import pandas as pd
import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras import backend as K
from keras.models import load_model
data = pd.read_csv("../input/fashion-mnist_train.csv")
test = pd.read_csv("../input/fashion-mnist_test.csv")
y_labels = np.array(data['label'].values)
y_labels_test = np.array(test['label'].values)
y_labels = np.eye(10)[y_labels.reshape(-1)]
print(y_labels.shape)
y_labels_test = np.eye(10)[y_labels_test.reshape(-1)]
print(y_labels_test.shape)
x_inputs = np.array(data.iloc[:,1:785].values)
x_inputs_test = np.array(test.iloc[:,1:785].values)
print(x_inputs.shape)
print(x_inputs_test.shape)
x_inputs = x_inputs.reshape(60000,28,28,1)
x_inputs_test = x_inputs_test.reshape(10000,28,28,1)
print(x_inputs.shape)
print(x_inputs_test.shape)
model = keras.Sequential()
model.add(Conv2D(32, (5, 5), input_shape=(28,28,1), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
model.fit(x_inputs, y_labels,
          batch_size=128,
          epochs=12,
          verbose=1,
          validation_data = (x_inputs_test,y_labels_test))
# Saving the model
model.save('my_model.h5')
score = model.evaluate(x_inputs_test, y_labels_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

x.shape
my_labels = ['T-shirt','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Boot']
item = 50
x = x_inputs_test[item].reshape(1,28,28,1)
print("Prediction: " + my_labels[np.argmax(model.predict(x))])
print("Original: " + my_labels[np.argmax(y_labels_test[item])])

