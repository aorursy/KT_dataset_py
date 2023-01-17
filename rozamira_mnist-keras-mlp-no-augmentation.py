#importing necessary libraries



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import matplotlib.cm as cm
#load the data

train_data = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")



#reshape the train and test data

X_train = (train_data.ix[:,1:].values).astype('float32')

#data labels

y_train = train_data.ix[:,0].values.astype('int32')

#test data

X_test = test_data.astype('float32')



print("The MNIST dataset has a training set of %d examples." % len(X_train))

print("The MNIST database has a test set of %d examples." % len(X_test))
#Convert training data to img format 

X_train = X_train.reshape(X_train.shape[0], 28, 28)

X_train.shape

X_test = X_test.values.reshape(-1, 28, 28,1)

X_test.shape

for i in range(6):

    plt.subplot(330 + (i+1))

    plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))

    plt.title(y_train[i]);
def visualize_input(img, ax):

    ax.imshow(img, cmap='gray')

    width, height = img.shape

    thresh = img.max()/2.5

    for x in range(width):

        for y in range(height):

            ax.annotate(str(round(img[x][y],2)), xy=(y,x),

                        horizontalalignment='center',

                        verticalalignment='center',

                        color='white' if img[x][y]<thresh else 'black')



fig = plt.figure(figsize = (12,12)) 

ax = fig.add_subplot(111)

visualize_input(X_train[5], ax)
#reshape the image and expand to 1 dimension

X_train = X_train.reshape(X_train.shape[0], 28, 28,1)

X_train.shape
X_test = X_test.reshape(X_test.shape[0], 28, 28,1)

X_test.shape
import seaborn as sns

sns.countplot(train_data['label'])
#normalize the image

X_train = X_train.astype('float32')/255

X_test = X_test.astype('float32')/255 
from keras.utils import np_utils

from sklearn.model_selection import train_test_split

# Set the random seed

random_seed = 2

# Split the train and the validation set for the fitting

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.15, random_state=random_seed)

# print first ten (integer-valued) training labels

print('Integer-valued labels:')

print(y_train[:10])



# one-hot encode the labels

y_train = np_utils.to_categorical(y_train, 10)

y_val = np_utils.to_categorical(y_val, 10)



# print first ten (one-hot) training labels

print('One-hot labels:')

print(y_train[:10])
from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Activation





model = Sequential()

model.add(Flatten(input_shape=X_train.shape[1:]))

model.add(Dense(512))

model.add(Activation('relu'))

model.add(Dropout(0.15))

model.add(Dense(256))

model.add(Activation('relu'))

model.add(Dropout(0.15))

model.add(Dense(10))

model.add(Activation('softmax'))



model.compile(loss='categorical_crossentropy',

              optimizer='rmsprop',

              metrics=['accuracy'])



model.summary()

print("input shape ",model.input_shape)

print("output shape ",model.output_shape)
#train the data

from keras.callbacks import ModelCheckpoint



batch_size = 128

nb_epoch = 20

checkpointer = ModelCheckpoint(filepath='mnist.model.best.hdf5', 

                               verbose=1, save_best_only=True)

hist = model.fit(X_train, y_train, epochs=nb_epoch, batch_size=batch_size, validation_data=(X_val,y_val), callbacks=[checkpointer],verbose=1, shuffle=True)
plt.title('Train Accuracy vs Val Accuracy')

plt.plot(hist.history['acc'], label='Train Accuracy', color='black')

plt.plot(hist.history['val_acc'], label='Validation Accuracy', color='red')

plt.legend()

plt.show()
# evaluate test accuracy



from keras.models import load_model



model = load_model('mnist.model.best.hdf5')

score = model.evaluate(X_val, y_val, verbose=0)

accuracy = 100*score[1]



# print test accuracy

print('Validation accuracy: %.4f%%' % accuracy)
predictions = model.predict_classes(X_test, verbose=1)



submissions = pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),

                            "Label": predictions})

submissions.to_csv("mnist_CNN_test.csv",index=False)