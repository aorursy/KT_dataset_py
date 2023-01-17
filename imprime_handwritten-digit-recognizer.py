import numpy as np

import matplotlib.pyplot as plt

import cv2

from keras.datasets import mnist

from keras.utils.np_utils import to_categorical

from keras.layers import Flatten, Dropout, Dense

from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.models import Sequential, Model

from keras.optimizers import Adam
np.random.seed(0)
import gzip

import sys

import pickle
f = gzip.open('../input/test-dataset/mnist.pkl.gz', 'rb')

if sys.version_info < (3,):

    data = pickle.load(f)

else:

    data = pickle.load(f, encoding='bytes')

f.close()

len(data[0])
(X_train, y_train),(X_test, y_test) = data
len(X_train)
len(X_test)
test_rand = np.random.randint(2,100)

plt.imshow(X_train[test_rand],cmap= 'gray')

y_train[test_rand]
print(X_train.shape)

print(X_test.shape)
plt.imshow(X_train[test_rand],cmap= 'gray')

print(X_train[test_rand])
num_of_samples=[]

cols = 5

num_classes = 10
fig, axs = plt.subplots(nrows= num_classes, ncols=cols, figsize= (5,10))

fig.tight_layout()

for i in range(cols):

  for j in range(num_classes):

    x_selected = X_train[y_train== j]

    axs[j][i].imshow(x_selected[np.random.randint(0,(len(x_selected)-1)), :, :], cmap=plt.get_cmap('gray'))

    axs[j][i].axis('off')

    if i == 2:

      axs[j][i].set_title(str(j))

      num_of_samples.append(len(x_selected))
print(num_of_samples)

plt.figure(figsize=(12,4))

plt.bar(range(num_classes), num_of_samples)

plt.title("Distribution of the train dataset")

plt.xlabel("Class number")

plt.ylabel("Number of images")

plt.show()
X_train.max()
X_train = X_train/255

X_test = X_test/255
X_train.shape
X_train = X_train.reshape(-1, 28, 28,1)

X_train.shape
X_test.shape
X_test= X_test.reshape(-1, 28, 28,1)
X_train.shape
y_cat_train = to_categorical(y_train)

y_cat_test = to_categorical(y_test)
y_cat_train.shape
y_test.shape
def le_net():

  model = Sequential()

  model.add(Conv2D(30,(5,5), input_shape=(28,28,1), activation = 'relu'))

  model.add(MaxPooling2D(pool_size=(2,2)))

  model.add(Conv2D(15,(3,3), activation='relu'))

  model.add(MaxPooling2D(pool_size=(2,2)))

  model.add(Flatten())

  model.add(Dense(500,activation="relu"))

  model.add(Dropout(0.5))

  model.add(Dropout(0.5))

  model.add(Dense(num_classes,activation="softmax"))

  model.compile(Adam(lr=0.001),loss ='categorical_crossentropy', metrics=['accuracy'])

  return model
lenet = le_net()

lenet.summary()
history= lenet.fit(X_train,y_cat_train, epochs=20, validation_split= 0.1, batch_size=400,verbose=1,shuffle=1)
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.legend(['loss','val_loss'])

plt.title('Loss')

plt.xlabel('epoch')
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.legend(['accuracy','val_accuracy'])

plt.title('Acc')

plt.xlabel('epoch')

lenet.metrics_names
y_cat_test = to_categorical(y_test)
lenet.evaluate(X_test,y_cat_test)
from sklearn.metrics import classification_report
y_predict = lenet.predict_classes(X_test)
y_predict
print(classification_report(y_test, y_predict))
from IPython.display import FileLink, FileLinks

FileLinks('.') #lists all downloadable files on server
# lenet.save('digit.h5')
!ls
# from google.colab import drive

# drive.mount('/content/drive')
# from google.colab import files
# files.download('digit.h5')
rand_num = np.random.randint(0,10000)

test_img = X_test[rand_num]

test_img= test_img.reshape(1,28,28,1)

print('predicted number is', lenet.predict_classes(test_img)[0])

print()



# checking the predicted num imgage

test_img= test_img.reshape(28,28)

plt.imshow(test_img, cmap='gray')
import requests

from PIL import Image

url = "https://kx.com/images/03_IMAGES/160520-8.png"

response = requests.get(url,stream=True)

img = Image.open(response.raw).convert("L")

plt.imshow(img,cmap="gray")
img_array = np.asarray(img)

# resizing to (28,28) img

res = cv2.resize(img_array,(28,28))

# pre-processing the Image

res = res/255

res = res.reshape(1,28,28,1)



# prdiction



print('predicted number is', lenet.predict_classes(res)[0])
