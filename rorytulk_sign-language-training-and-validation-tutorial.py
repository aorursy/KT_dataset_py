import os
import cv2
from matplotlib import pyplot as plt

import numpy as np
import keras
from keras.models import Sequential
import keras.layers as layers
from keras import optimizers

import sklearn.model_selection as model_selection


#TODO load the X and Y dataset that we have saved from process_images

dataset_path = '../input/sign-language-digits-dataset/Sign-language-digits-dataset'

arrays_dataset_path = os.path.join(dataset_path, 'Arrays')

X = None
Y = None
print('X shape : {}  Y shape: {}'.format(X.shape, Y.shape))
plt.imshow(X[700], cmap='gray')
print(Y[700]) # one-hot labels starting at zero
#TODO: implement training and validation dataset split

def split_data(X, Y, validation_size):
    return None

Xtrain, Xtest, Ytrain, Ytest = split_data(X, Y, 0.2)
print('Xtrain shape {} Ytrain shape {}'.format(Xtrain.shape, Ytrain.shape))
print('Xtest shape {} Ytest shape {}'.format(Xtest.shape, Ytest.shape))
#TODO: the first Conv2D layer needs to specify what it takes as an input. Since we have resized all the images to
#a specific size, please specify the input_shape

#TODO: print out information about the model to help visualize how layers are structured

model = Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=None)) 
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu')) 
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.not_implemented()

model.compile(loss='categorical_crossentropy',
             optimizer=optimizers.adadelta(),
             metrics=['accuracy'])
#TODO: implement add_channel_dim so that it adds a new dimension. Look at reshape and newaxis

def add_channel_dim(X):
    return None

Xtrain_batch = add_channel_dim(Xtrain)

Xtest_batch = add_channel_dim(Xtest)
#train our model
history = model.fit(Xtrain_batch, Ytrain, batch_size=32, epochs=9, validation_data=(Xtest_batch, Ytest))
acc = history.history['acc']
val_acc = history.history['val_acc']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
#TODO: evaluate the model
eval_score = None

print('Evaluation score {}'.format(eval_score))
def display_img(img_path):
    img = cv2.imread(img_path)
    color_corrected = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(color_corrected)
    plt.title(img_path)
    plt.show()

#TODO: use the same function that you've implemented in process_image
# resize to 64 x 64 and greyscale
    
def get_gsimg(image_path):
    img = cv2.imread(image_path)
    resize_img = cv2.resize(img, (64, 64))
    gs_img = cv2.cvtColor(resize_img, cv2.COLOR_BGR2GRAY)
    return gs_img

#prediction
dataset_path='../input/sign-language-dataset/sign_lang_dataset/sign_lang_dataset'

input_path = os.path.join(dataset_path, 'Inputs')

display_img(os.path.join(input_path, 'Sample_1.jpg'))

sample1 = get_gsimg(os.path.join(input_path, 'Sample_1.jpg'))
sample1_batch = add_channel_dim(np.array(sample1).reshape((1, 64, 64)))
model.predict(sample1_batch)
display_img(os.path.join(input_path, 'Sample_3.jpg'))

sample3 = get_gsimg(os.path.join(input_path, 'Sample_3.jpg'))
sample3_batch = add_channel_dim(np.array(sample3).reshape((1, 64, 64)))

model.predict(sample3_batch)