import pandas as pd

import tensorflow as tf

from tensorflow.keras import Sequential

from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, Conv2D, MaxPool2D

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.preprocessing import image

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import time

from sklearn import metrics

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from tqdm import tqdm

import os
labels_train = pd.read_csv('../input/super-ai-image-classification/train/train/train.csv')

labels_val = pd.read_csv('../input/super-ai-image-classification/val/val/val.csv')
print(labels_train.shape)

labels_train.head()
labels_val.shape
img_width = 350

img_height = 350

X = []

for i in tqdm(range(labels_train.shape[0])):

  path = '../input/super-ai-image-classification/train/train/images/' + labels_train['id'][i]

  img = image.load_img(path, target_size = (img_width, img_height, 3))

  img = image.img_to_array(img)

  img = img/255.0

  X.append(img)



X = np.array(X)
y = labels_train.drop(['id'], axis = 1)

y = y.to_numpy()

y.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size = 0.2)
kf = KFold(5)

num_epoch = 30

t0 = time.time()

kf = KFold(n_splits=3, random_state=100)



oos_y = []

oos_prd = []

fold = 0



for train, test in kf.split(X):

  fold += 1

  print('Fold #{}'.format(fold))



  model = Sequential()

  model.add(Conv2D(16,(3,3), activation = 'relu', input_shape = X_train[0].shape))

  model.add(BatchNormalization())

  model.add(MaxPool2D(2,2))

  model.add(Dropout(0.3))



  model.add(Conv2D(32,(3,3), activation = 'relu'))

  model.add(BatchNormalization())

  model.add(MaxPool2D(2,2))

  model.add(Dropout(0.3))



  model.add(Conv2D(64,(3,3), activation = 'relu'))

  model.add(BatchNormalization())

  model.add(MaxPool2D(2,2))

  model.add(Dropout(0.4))



  model.add(Conv2D(128,(3,3), activation = 'relu'))

  model.add(BatchNormalization())

  model.add(MaxPool2D(2,2))

  model.add(Dropout(0.4))



  model.add(Flatten())



  model.add(Dense(128, activation = 'relu'))

  model.add(BatchNormalization())

  model.add(Dropout(0.5))



  model.add(Dense(128, activation = 'relu'))

  model.add(BatchNormalization())

  model.add(Dropout(0.5))



  model.add(Dense(1, activation = 'sigmoid'))



  model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



  history = model.fit(X_train, y_train, epochs = num_epoch, validation_data = (X_test, y_test))



  pred = model.predict(X_test)



  oos_y.append(y_test)

  pred = np.argmax(pred, axis=1)

  oos_prd.append(pred)



  y_compare = np.argmax(y_test, axis=1)

  score = metrics.accuracy_score(y_compare, pred)

  print('Folder score (accuracy): {}'.format(score))
def plot_learningCurve(history, epoch):

  plt.figure()

  plt.plot(np.arange(0, num_epoch), history.history['accuracy'])

  plt.plot(np.arange(0, num_epoch), history.history['val_accuracy'])

  plt.title('Model accuracy')

  plt.ylabel('Accuracy')

  plt.xlabel('Epoch')

  plt.legend(['Train', 'Val'], loc = 'upper left')

  plt.show()



  plt.figure()

  plt.plot(np.arange(0, num_epoch), history.history['loss'])

  plt.plot(np.arange(0, num_epoch), history.history['val_loss'])

  plt.title('Model loss')

  plt.ylabel('Loss')

  plt.xlabel('Epoch')

  plt.legend(['Train', 'Val'], loc = 'upper left')

  plt.show()
plot_learningCurve(history, num_epoch)
img = image.load_img('../input/super-ai-image-classification/val/val/images/000af6a7-8302-4085-b448-6eae2c1a60e5.jpg', target_size = (img_width, img_height, 3))

plt.imshow(img)

img = image.img_to_array(img)

img = img/255.0

img = img.reshape(1, img_width, img_height, 3)

classes = labels_val['category']

y_prob = model.predict(img)

top = np.argsort(y_prob[0])

print(top[0])

labels_val['category'][0]
img_list = os.listdir("../input/super-ai-image-classification/val/val/images")

name = []

predict = []



for i in img_list:

    img = image.load_img('../input/super-ai-image-classification/val/val/images/'+i, target_size = (img_width, img_height, 3))

    img = image.img_to_array(img)

    img = img/255.0

    img = img.reshape(1, img_width, img_height, 3)

    classes = labels_val['category']

    y_prob = model.predict(img)

    top = np.argsort(y_prob[0])

    name.append(i)

    predict.append(top[0])
data = {'id':name, 'category':predict}

df = pd.DataFrame(data) 
df
df.to_csv('22p21s0125.csv', index=False)
out_put = pd.read_csv('./22p21s0125.csv')
out_put.tail()