import pandas as pd

import numpy as np

import os

from IPython.display import Image, display_png

import matplotlib.pyplot as plt

import PIL as pil

import random

from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D,MaxPooling2D
print(os.listdir('../input'))
d= os.listdir('../input/kkanji')

print('Directories: ' + str(d[:10]))
dr = '../input/kkanji/kkanji2/'

d= os.listdir(dr)

print('Directories: ' + str(d[:10]))
print(os.listdir(dr + 'U+8AA4')[:10])
dict = {'U+4E00': '1',

        'U+4E8C': '2',

        'U+4E09': '3', 

        'U+56DB': '4', 

        'U+4E94': '5', 

        'U+516D': '6', 

        'U+4E03': '7' ,

        'U+516B': '8', 

        'U+4E5D': '9', 

        'U+5341': '10', 

        'U+767E': '100', 

        'U+5343': '1000',

        'U+4E07': '10000' }



pngs ={}



for kanji_dir in dict.keys():

    d2 =dr + kanji_dir

    

    if (os.path.isdir(d2)):

        png_list = os.listdir(d2)

        pngs[kanji_dir] = png_list

        

        print("Number: " + dict.get(kanji_dir)+ " , " + str(len(png_list))+ " samples")

        

        fig, ax = plt.subplots(nrows=1, ncols=5)

        

        for i in range(5):

            img = np.array(pil.Image.open(d2+'/'+ png_list[i]))

            ax[i].imshow(img, cmap = 'gray')

        ax[0].set_xticks([])

        ax[0].set_yticks([])

        plt.tight_layout()

        plt.show()
samples = 115

random.seed(1)



pngs2={}



for d in dict.keys():

    pngs2[d] = random.sample(pngs[d], samples)
img_shape = 64 * 64



X_samples =np.empty((0,img_shape), int)

y_samples =np.array([],int)

i=0



for d in dict.keys():

    d2 =dr+d

    for f in pngs2[d]:

        img = np.array(pil.Image.open(d2 +'/'+ f))

        X_samples = np.append(X_samples, img.reshape(1,img_shape), axis=0)

        y_samples = np.append(y_samples, i)

    i=i+1
(X_train, X_test, y_train, y_test) = train_test_split(X_samples, y_samples, test_size=0.1, random_state=0)



y_train_onehot = np.eye(13)[y_train]



mean_vals = np.mean(X_train, axis=0)

std_val = np.std(X_train)



X_train_centered = (X_train - mean_vals) / std_val

X_test_centered = (X_test - mean_vals) / std_val
img_rows , img_cols = 64,64

batch_size = 64

num_classes = 13

epochs = 50



model = tf.keras.models.Sequential()



model.add(Conv2D(32, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=(img_rows, img_cols, 1)))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))



model.compile(loss = tf.keras.losses.categorical_crossentropy, 

                             optimizer = tf.keras.optimizers.Adam(),

                                 metrics = ['accuracy'])
X_train_centered_reshape = X_train_centered.reshape(X_train_centered.shape[0], 64, 64, 1)

history = model.fit(X_train_centered_reshape,y_train_onehot, batch_size = batch_size, epochs =epochs, verbose = 1, validation_split = 0.3)
plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.legend(['Training', 'Validation'])

plt.title('Accuracy')

plt.xlabel('Epochs')
X_test_centered_reshape = X_test_centered.reshape(X_test_centered.shape[0], 64, 64, 1)

y_test_onehot = np.eye(13)[y_test]



test_accuracy = model.evaluate(X_test_centered_reshape, y_test_onehot)

print("Test set accuracy: "+ str(test_accuracy[1]) + " , loss: " + str(test_accuracy[0]))
dict2 ={0:"1",

        1:"2",

        2:"3",

        3:"4",

        4:"5",

        5:"6",

        6:"7",

        7:"8",

        8:"9",

        9:"10",

        10:"100",

        11:"1000",

        12:"10000"}



y_test_pred = model.predict_classes(X_test_centered_reshape)



print ("----- Wrong prediction in test set ------")



for label, pred,i in zip(y_test, y_test_pred, range(0,len(y_test))):

    if (label != pred):

        print("Label: "+ dict2.get(label)+ ", Prediction: " + dict2.get(pred)) 

        plt.imshow(X_test[i].reshape(64,64),cmap = 'gray')

        plt.show()

        

    

    