import numpy as np         

import pandas as pd

import matplotlib.pyplot as plt      

import random                        



from keras.datasets import mnist     

from keras.models import Sequential  



from keras.layers.core import Dense, Dropout, Activation 

from keras.utils import np_utils                         
(X_train, y_train), (X_test, y_test) = mnist.load_data()



print("X_train shape", X_train.shape)

print("y_train shape", y_train.shape)

print("X_test shape", X_test.shape)

print("y_test shape", y_test.shape)
test_data = pd.read_csv('/kaggle/input/digit-recognizer/test.csv', delimiter = ',', header = 0, usecols = [x for x in range(0, 784)])
for i in range(9):

    plt.subplot(3,3,i+1)

    num = random.randint(0, len(X_train))

    plt.imshow(X_train[num], cmap='gray', interpolation='none')

    plt.title("Class {}".format(y_train[num]))

    

plt.tight_layout()
X_train = X_train.reshape(60000, 784) 

X_test = X_test.reshape(10000, 784)   



X_train = X_train.astype('float32')   

X_test = X_test.astype('float32')



X_train /= 255                        

X_test /= 255



print("Training matrix shape", X_train.shape)

print("Testing matrix shape", X_test.shape)
no_classes = 10



Y_train = np_utils.to_categorical(y_train, no_classes)

Y_test = np_utils.to_categorical(y_test, no_classes)
model = Sequential()
model.add(Dense(512, input_shape=(784,))) 
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))

model.add(Activation('relu'))

model.add(Dropout(0.2))
model.add(Dense(10))

model.add(Activation('softmax'))
model.summary()
from keras.utils import plot_model

plot_model(model, to_file='model_chart.png', show_shapes=True, show_layer_names=True)

from IPython.display import Image

Image("model_chart.png")
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, Y_train,

          batch_size=128, epochs=10,

          verbose=1)
score = model.evaluate(X_test, Y_test)

print('Test accuracy:', score[1])
fig = plt.figure()

plt.subplot(2,1,1)

plt.plot(history.history['accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='lower right')



plt.subplot(2,1,2)

plt.plot(history.history['loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper right')



plt.tight_layout()
results = model.predict(test_data)
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("submission.csv",index=False)
submission
predicted_classes = model.predict_classes(X_test)



correct_indices = np.nonzero(predicted_classes == y_test)[0]



incorrect_indices = np.nonzero(predicted_classes != y_test)[0]
plt.figure()

for i, correct in enumerate(correct_indices[:9]):

    plt.subplot(3,3,i+1)

    plt.imshow(X_test[correct].reshape(28,28), cmap='gray', interpolation='none')

    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], y_test[correct]))

    

plt.tight_layout()

    

plt.figure()

for i, incorrect in enumerate(incorrect_indices[:9]):

    plt.subplot(3,3,i+1)

    plt.imshow(X_test[incorrect].reshape(28,28), cmap='gray', interpolation='none')

    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], y_test[incorrect]))

    

plt.tight_layout()