import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split



import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator

from keras.utils import to_categorical



from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout

from keras.layers import Dense, Flatten



import cv2
import os

print(os.listdir('../input'))
category = ["cat", "dog"]



EPOCHS                  = 50

IMGSIZE                 = 128

BATCH_SIZE              = 32

STOPPING_PATIENCE       = 15

VERBOSE                 = 1

MODEL_NAME              = 'cnn_50epochs_imgsize128'

OPTIMIZER               = 'adam'

TRAINING_DIR            = '../input/train'

TEST_DIR                = '../input/test'
for img in os.listdir(TRAINING_DIR)[7890:]:

    img_path = os.path.join(TRAINING_DIR, img)

    img_arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    img_arr = cv2.resize(img_arr, (IMGSIZE, IMGSIZE))

    plt.imshow(img_arr, cmap='gray')

    plt.title(img.split('.')[0])

    break
def create_train_data(path):

    X = []

    y = []

    for img in os.listdir(path):

        if img == os.listdir(path)[7889]:

            continue

        img_path = os.path.join(path, img)

        img_arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        img_arr = cv2.resize(img_arr, (IMGSIZE, IMGSIZE))

        img_arr = img_arr / 255.0

        cat = np.where(img.split('.')[0] == 'dog', 1, 0)

        

        X.append(img_arr)

        y.append(cat)

    

    X = np.array(X).reshape(-1, IMGSIZE, IMGSIZE, 1)

    y = np.array(y)

    

    return X, y        
X, y = create_train_data(TRAINING_DIR)



print(f"features shape {X.shape}.\nlabel shape {y.shape}.")
y = to_categorical(y, 2)

print(f"features shape {X.shape}.\nlabel shape {y.shape}.")
X_train , X_test, y_train, y_test = train_test_split(X, y, test_size=1/3)
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2,horizontal_flip=True)

train_gen = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)



test_datagen = ImageDataGenerator(rescale=1./255)

test_gen = train_datagen.flow(X_test, y_test, batch_size=BATCH_SIZE)
model = Sequential()



model.add(Conv2D(32, (3, 3), activation='relu', input_shape=X.shape[1:]))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.5))



model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.5))



model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.5))



model.add(Conv2D(256, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.5))



model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.5))



model.add(Dense(2, activation='sigmoid'))



model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=1/3)
"""history = model.fit_generator(train_gen, 

                              epochs=EPOCHS, 

                              steps_per_epoch=len(X_train)/BATCH_SIZE, 

                              validation_data=test_gen, 

                              validation_steps=len(X_test)/BATCH_SIZE)

                              """
model.save_weights("CATSvsDOGS_model.h5")

model.save('CNN_CAT.model')
train_acc = model.evaluate(X_train, y_train, batch_size=BATCH_SIZE)

test_acc = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10))

ax1.plot(history.history['loss'], color='b', label="Training loss : {:0.4f}".format(train_acc[0]))

ax1.plot(history.history['val_loss'], color='r', label="validation loss : {:0.4f}".format(test_acc[0]))

ax1.set_xticks(np.arange(1, EPOCHS, 1))

ax1.set_yticks(np.arange(0, 1., 0.1))

ax1.legend()



ax2.plot(history.history['acc'], color='b', label="Training accuracy : {0:.4f}".format(train_acc[1]))

ax2.plot(history.history['val_acc'], color='r',label="Validation accuracy : {0:.4f}".format(test_acc[1]))

ax2.set_xticks(np.arange(1, EPOCHS, 1))

ax2.set_yticks(np.arange(0.4, 1.2, 0.1))



legend = plt.legend(loc='best', shadow=True)

plt.tight_layout()

plt.show()
im_test = []

c = 0

for img in os.listdir(TEST_DIR)[:50]:

    c = c + 1

    img_path = os.path.join(TEST_DIR, img)

    img_arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    img_arr = cv2.resize(img_arr, (IMGSIZE, IMGSIZE))

    img_arr = img_arr / 255.0

    im_test.append(img_arr)

print(c) 

im_test = np.array(im_test).reshape(-1, IMGSIZE, IMGSIZE, 1)

im_pred = model.predict(im_test)
fig , ax = plt.subplots(3, 3, figsize=(30, 25))

for i, axis in enumerate(ax.flat):

    axis.imshow(im_test[i].reshape(128, 128), cmap='gray')

    #axis.set(title=f'{im_pred[i].max()} => {category[im_pred[i].argmax()]}')

    axis.set_title(f'Predict: {im_pred[i].max()} => {category[im_pred[i].argmax()]}', fontsize=20)
model.summary()