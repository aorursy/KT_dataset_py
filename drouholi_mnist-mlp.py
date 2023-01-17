import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import matplotlib.cm as cm
train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")
X_train = (train_data.ix[:,1:].values).astype('float32')
y_train = train_data.ix[:,0].values.astype('int32')
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
X_train
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
#reshape/expand 1 dimention 
X_train = X_train.reshape(X_train.shape[0], 28, 28,1)
X_train.shape
X_test = X_test.reshape(X_test.shape[0], 28, 28,1)
X_test.shape
X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255 
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
# Set the random seed
random_seed = 2
# Split the train and the validation set for the fitting
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state=random_seed)
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
from keras.layers import Dense, Dropout, Flatten

# define the model
model = Sequential()
model.add(Flatten(input_shape=X_train.shape[1:]))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

# summarize the model
model.summary()
print("input shape ",model.input_shape)
print("output shape ",model.output_shape)
# compile the model
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', 
              metrics=['accuracy'])
# evaluate test accuracy
score = model.evaluate(X_val, y_val, verbose=0)
accuracy = 100*score[1]

# print test accuracy
print('validation accuracy: %.4f%%' % accuracy)
from keras.callbacks import ModelCheckpoint   

# train the model
checkpointer = ModelCheckpoint(filepath='mnist.model.best.hdf5', 
                               verbose=1, save_best_only=True)
hist = model.fit(X_train, y_train, batch_size=128, epochs=20,
          validation_split=0.2, callbacks=[checkpointer],
          verbose=1, shuffle=True)
# evaluate test accuracy
score = model.evaluate(X_val, y_val, verbose=0)
accuracy = 100*score[1]

# print test accuracy
print('validation accuracy: %.4f%%' % accuracy)
#let's try data augmentation to reduce the over fitting on training data 
from keras.preprocessing.image import ImageDataGenerator
#don't apply a vertical_flip nor horizontal_flip since 
#it could lead to misclassify symetrical numbers such as 6 and 9.
dataAug = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images
dataAug.fit(X_train)
hist = model.fit_generator(dataAug.flow(X_train, y_train, batch_size=128), epochs=20,
          validation_data = (X_val,y_val), callbacks=[checkpointer],
          verbose=2)
# evaluate test accuracy
score = model.evaluate(X_val, y_val, verbose=0)
accuracy = 100*score[1]

# print test accuracy
print('validation accuracy: %.4f%%' % accuracy)
results = model.predict(X_test)
results = np.argmax(results,axis = 1)
results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("mnist_CNN_test.csv",index=False)

