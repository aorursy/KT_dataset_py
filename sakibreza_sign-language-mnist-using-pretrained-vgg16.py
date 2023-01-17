from IPython.display import Image

Image("../input/sign-language-mnist/amer_sign2.png")
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv('../input/sign-language-mnist/sign_mnist_train.csv')

test = pd.read_csv('../input/sign-language-mnist/sign_mnist_test.csv')
train.head()
train.shape
Image("../input/sign-language-mnist/american_sign_language.PNG")
labels = train['label'].values
unique_val = np.array(labels)

np.unique(unique_val)
plt.figure(figsize = (18,8))

sns.countplot(x =labels)
train.drop('label', axis = 1, inplace = True)
images = train.values

images = np.array([np.reshape(i, (28, 28)) for i in images])

images = np.array([i.flatten() for i in images])
from sklearn.preprocessing import LabelBinarizer

label_binrizer = LabelBinarizer()

labels = label_binrizer.fit_transform(labels)
labels
plt.imshow(images[0].reshape(28,28))
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size = 0.3, random_state = 101)
import keras

from keras.models import Sequential

from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

from keras.applications import VGG16;

from keras.applications.vgg16 import preprocess_input
batch_size = 128

num_classes = 24

epochs = 50
x_train = x_train / 255

x_test = x_test / 255
x_train_t = np.stack([x_train.reshape(x_train.shape[0],28,28)]*3, axis=3).reshape(x_train.shape[0],28,28,3)

x_test_t = np.stack([x_test.reshape(x_test.shape[0],28,28)]*3, axis=3).reshape(x_test.shape[0],28,28,3)

x_train_t.shape, x_test_t.shape
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
plt.imshow(x_train_t[0].reshape(28,28,3))
# model = Sequential()

# model.add(Conv2D(64, kernel_size=(3,3), activation = 'relu', input_shape=(28, 28 ,1) ))

# model.add(MaxPooling2D(pool_size = (2, 2)))



# model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))

# model.add(MaxPooling2D(pool_size = (2, 2)))



# model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))

# model.add(MaxPooling2D(pool_size = (2, 2)))



# model.add(Flatten())

# model.add(Dense(128, activation = 'relu'))

# model.add(Dropout(0.20))

# model.add(Dense(num_classes, activation = 'softmax'))

# model.summary()
# Resize the images 48*48 as required by VGG16

from keras.preprocessing.image import img_to_array, array_to_img

x_train_tt = np.asarray([img_to_array(array_to_img(im, scale=True).resize((48,48))) for im in x_train_t])/225

x_test_tt = np.asarray([img_to_array(array_to_img(im, scale=True).resize((48,48))) for im in x_test_t])/225

x_train_tt.shape, x_test_tt.shape
plt.imshow(x_test_tt[0].reshape(48,48,3))
model = Sequential()

#  Create base model of VGG16

model.add(VGG16(weights='../input/vgg16-pretrained/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',

                  include_top=False, pooling = 'avg',  

                  input_shape=(48, 48, 3)

                 ))

# 2nd layer as Dense 

model.add(Dense(num_classes, activation = 'softmax'))



# Say not to train first layer (ResNet) model as it is already trained

model.layers[0].trainable = False

model.summary()
# model = Sequential()

# model.add(VGG16(weights='../input/vgg16-pretrained/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',

#                   include_top=False, pooling = 'avg',  

#                   input_shape=(48, 48, 3))

# model.add(Dense(512, activation='relu', input_dim=input_shape))

# model.add(Dropout(0.3))

# model.add(Dense(512, activation='relu'))

# model.add(Dropout(0.3))

# model.add(Dense(1, activation='sigmoid')) 

# model.summary()
# model.compile(loss='binary_crossentropy',

#               optimizer=optimizers.RMSprop(lr=1e-5),

#               metrics=['accuracy'])
model.compile(loss = keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),

              metrics=['accuracy'])
history = model.fit(x_train_tt, y_train, validation_data = (x_test_tt, y_test), epochs=epochs, batch_size=batch_size)
plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title("Accuracy")

plt.xlabel('epoch')

plt.ylabel('accuracy')

plt.legend(['train','test'])

plt.show()
test_labels = test['label']
test.drop('label', axis = 1, inplace = True)
test_images = test.values

test_images = np.array([np.reshape(i, (28, 28)) for i in test_images])

#test_images = np.array([i.flatten() for i in test_images])

test_images_t = np.stack([test_images.reshape(test_images.shape[0],28,28)]*3, axis=3).reshape(test_images.shape[0],28,28,3)



# Resize the images 48*48 as required by VGG16

from keras.preprocessing.image import img_to_array, array_to_img

test_images_tt = np.asarray([img_to_array(array_to_img(im, scale=True).resize((48,48))) for im in test_images_t])/225

test_images_tt.shape
plt.imshow(test_images_tt[0].reshape(48,48,3))
test_labels = label_binrizer.fit_transform(test_labels)
test_images = test_images_tt.reshape(test_images.shape[0], 48, 48, 3)
test_images.shape
y_pred = model.predict(test_images)
from sklearn.metrics import accuracy_score
accuracy_score(test_labels, y_pred.round())