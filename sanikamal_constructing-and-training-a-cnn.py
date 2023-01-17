import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
from scipy import signal
i = np.random.randint(x_train.shape[0])



c = x_train[i,:,:]
plt.imshow(c,cmap='gray'); plt.axis('off');
plt.title('original image');
plt.figure(figsize=(18,8))
for i in range(10):
    k = -1.0 + 1.0*np.random.rand(3,3)
    c_digit = signal.convolve2d(c, k, boundary='symm', mode='same');
    plt.subplot(2,5,i+1);
    plt.imshow(c_digit,cmap='gray'); plt.axis('off');


batch_size = 128
nb_classes = 10
nb_epoch = 6

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

#sub-sample of test data to improve training speed. Comment out 
#if you want to train on full dataset.
x_train = x_train[:20000,:,:,:]
y_train = y_train[:20000]


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_test_inds = y_test.copy()
y_train_inds = y_train.copy()
y_train = keras.utils.to_categorical(y_train, nb_classes)
y_test = keras.utils.to_categorical(y_test, nb_classes)
#Create sequential convolutional multi-layer perceptron with max pooling and dropout
#uncomment if you want to add more layers (in the interest of time we use a shallower model)
model = Sequential()
model.add(Conv2D(nb_filters, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape)) #nb_filters, 
#model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
#model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
#Let's see what we've constructed layer by layer
model.summary()
model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epoch,
          verbose=1, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
#choose a random data from test set and show probabilities for each class.
i = np.random.randint(0,len(x_test))
digit = x_test[i].reshape(28,28)

plt.figure();
plt.subplot(1,2,1);
plt.title('Example of digit: {}'.format(y_test_inds[i]));
plt.imshow(digit,cmap='gray'); plt.axis('off');
probs = model.predict_proba(digit.reshape(1,28,28,1),batch_size=1)
plt.subplot(1,2,2);
plt.title('Probabilities for each digit class');
plt.bar(np.arange(10),probs.reshape(10),align='center'); plt.xticks(np.arange(10),np.arange(10).astype(str));
predictions = model.predict_classes(x_test, batch_size=32, verbose=1)
inds = np.arange(len(predictions))
wrong_results = inds[y_test_inds!=predictions]
#choose a random wrong result from the test set
i = np.random.randint(0,len(wrong_results))
i = wrong_results[i]
digit = x_test[i].reshape(28,28)

plt.figure();

plt.subplot(1,2,1);
plt.title('Digit {}'.format(y_test_inds[i]));
plt.imshow(digit,cmap='gray'); plt.axis('off');
probs = model.predict_proba(digit.reshape(1,28,28,1),batch_size=1)
plt.subplot(1,2,2);
plt.title('Digit classification probability');
plt.bar(np.arange(10),probs.reshape(10),align='center'); plt.xticks(np.arange(10),np.arange(10).astype(str));
prediction_probs = model.predict_proba(x_test, batch_size=32, verbose=1)
wrong_probs = np.array([prediction_probs[ind][digit] for ind,digit in zip(wrong_results,predictions[wrong_results])])
all_probs =  np.array([prediction_probs[ind][digit] for ind,digit in zip(np.arange(len(predictions)),predictions)])
#plot as histogram
plt.hist(wrong_probs,alpha=0.5,density=True,label='wrongly-labeled');
plt.hist(all_probs,alpha=0.5,density=True,label='all labels');
plt.legend();
plt.title('Comparison between wrong and correctly classified labels');
plt.xlabel('highest probability');
print (model.layers[0].get_weights()[0].shape)
weights = model.layers[0].get_weights()[0]
for i in range(nb_filters):
    plt.subplot(6,6,i+1)
    plt.imshow(weights[:,:,0,i],cmap='gray',interpolation='none'); plt.axis('off');
#Create new sequential model, same as before but just keep the convolutional layer. 
model_new = Sequential()
model_new.add(Conv2D(nb_filters, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape)) 
#set weights for new model from weights trained on MNIST.
for i in range(1):
    model_new.layers[i].set_weights(model.layers[i].get_weights())
#pick a random digit and "predict" on this digit (output will be first layer of CNN)
i = np.random.randint(0,len(x_test))
digit = x_test[i].reshape(1,28,28,1)
pred = model_new.predict(digit)
#check shape of prediction
print (pred.shape)
#For all the filters, plot the output of the input
plt.figure(figsize=(18,18))
filts = pred[0]
for i in range(nb_filters):
    filter_digit = filts[:,:,i]
    plt.subplot(6,6,i+1)
    plt.imshow(filter_digit,cmap='gray'); plt.axis('off');