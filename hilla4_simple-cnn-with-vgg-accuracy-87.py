import os 

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt 

from keras import models

from keras import layers

from keras import optimizers

from keras.preprocessing import image

from keras.applications import VGG16
base_dir = '/kaggle/input/dogs-vs-cats-redux-kernels-edition/'

train_dir = os.path.join(base_dir, 'train/')

test_dir = os.path.join(base_dir, 'test/')
train_images = []

for i in os.listdir(train_dir):

    train_images.append(train_dir+i)

    

test_images = []

for i in os.listdir(test_dir):

    test_images.append(test_dir+i)
conv_m = VGG16(weights='imagenet',

               include_top=False,

               input_shape=(150,150,3))
conv_m.summary()
model = models.Sequential()

model.add(conv_m)

model.add(layers.Flatten())

model.add(layers.Dense(256, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))
model.summary()
conv_m.trainable = False
model.compile(loss='binary_crossentropy',

             optimizer=optimizers.RMSprop(lr=2e-5),

             metrics=['acc'])
train = train_images[:2000]

train_numpy = []

for i in train:

    img = image.load_img(i, target_size=(150,150))

    img = image.img_to_array(img)

    img = img/255

    train_numpy.append(img)

    

train_numpy = np.array(train_numpy)
train_y = []

for i in train:

    if 'dog.' in i:

        train_y.append(1)

    else:

        train_y.append(0)
history = model.fit(train_numpy, train_y, batch_size=100, epochs=20, validation_split=0.25)
loss = history.history['loss']

val_loss = history.history['val_loss']

acc = history.history['acc']

val_acc = history.history['val_acc']

epochs = range(1, len(loss)+1)
plt.plot(epochs, loss, 'bo', label='Loss')

plt.plot(epochs, val_loss, 'b', label='Val_loss')

plt.title("Loss")

plt.legend()

plt.show()
plt.plot(epochs, acc, 'bo', label='Acc')

plt.plot(epochs, val_acc, 'b', label='Val_Acc')

plt.title("Acc")

plt.legend()

plt.show()
pred = model.predict(train_numpy[1:2])
if pred >= 0.5: 

    print('I am {:.2%} sure this is a Dog'.format(pred[0][0]))

else: 

    print('I am {:.2%} sure this is a Cat'.format((1-pred[0])[0]))
image.load_img(train[1])


test_numpy = []

for i in test_images:

    if '/kaggle/input/dogs-vs-cats-redux-kernels-edition/test/test' in i :

        continue

    test_img = image.load_img(i, target_size=(150,150))

    test_img = image.img_to_array(test_img)

    test_img = test_img/255

    test_numpy.append(test_img)

    

test_numpy = np.array(test_numpy)
test_answer = model.predict(test_numpy)
test_id = []

for i in os.listdir(test_dir):

    if 'test' in i :

        continue    

    num = i.split('.')[0]

    test_id.append(num)
test_id_sub = pd.Series(test_id, name='id')
results = pd.Series(test_answer.reshape(12500,), name='label')
submission = pd.concat([test_id_sub, results], axis=1)
submission
submission.to_csv("Cats_and_Dogs_CNN_VGG.csv", index=False)