!date
!pip install split-folders tqdm
from keras.models import Sequential

from keras.layers import Conv2D, Flatten, Dense,MaxPooling2D, Dropout

import numpy as np

from keras.preprocessing import image

from keras.callbacks import EarlyStopping, ModelCheckpoint

import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

%matplotlib inline
try:

    from tensorflow.python.util import module_wrapper as deprecation

except ImportError:

    from tensorflow.python.util import deprecation_wrapper as deprecation

deprecation._PER_MODULE_WARNING_LIMIT = 0
DATASET_DIR = '/kaggle/input/real-and-fake-face-detection/real_and_fake_face_detection/real_and_fake_face'
import splitfolders



# Split with a ratio.

# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.

splitfolders.ratio(DATASET_DIR, output="/kaggle/working/data", seed=1337, ratio=(.8, .1, .1)) # default values

nbatch = 32

train_datagen = ImageDataGenerator(

    rescale=1./255,

    rotation_range=10.,

    width_shift_range=0.1,

    height_shift_range=0.1,

    zoom_range=0.2,

    horizontal_flip=True)



test_datagen = ImageDataGenerator(rescale = 1./255)



training_set = train_datagen.flow_from_directory('/kaggle/working/data/train',

                                                 target_size=(128,128),

                                                 batch_size =nbatch,

                                                 class_mode = 'binary')



val_set = test_datagen.flow_from_directory('/kaggle/working/data/val',

                                            target_size=(128,128),

                                            batch_size =nbatch,

                                            class_mode = 'binary')







test_set = test_datagen.flow_from_directory('/kaggle/working/data/test',

                                            target_size=(128,128),

                                            batch_size =nbatch,

                                            class_mode = 'binary')
h1 = plt.hist(training_set.classes, bins=range(0,3), alpha=0.8, color='blue', edgecolor='black')

h2 = plt.hist(val_set.classes,  bins=range(0,3), alpha=0.8, color='red', edgecolor='black')

plt.ylabel('# of instances')

plt.xlabel('Class')
for X, y in training_set:

    print(X.shape, y.shape)

    plt.figure(figsize=(16,16))

    for i in range(16):

        plt.subplot(4,4,i+1)

        plt.axis('off')

        plt.title('Label: ')

        img = np.uint8(255*X[i,:,:,0])

        plt.imshow(img, cmap='gray')

    break
model = Sequential()



model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(128,128,3)))

model.add(MaxPooling2D(pool_size = (2, 2))) 

model.add(Dropout(0.1))

model.add(Conv2D(32, (3, 3), activation = 'relu'))  

model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Dropout(0.2))

model.add(Conv2D(32, (3, 3), activation = 'relu'))  

model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(units = 128, activation = 'relu'))

model.add(Dense(units = 1, activation = 'sigmoid')) #'tanh'))



model.summary()
model.compile(optimizer = 'adam',

              loss = 'binary_crossentropy',

              metrics = ['accuracy'])
callbacks_list = [

    EarlyStopping(monitor='val_loss', patience=10),

    ModelCheckpoint(filepath='model_checkpoint.hdf5', monitor='val_loss', save_best_only=True, mode ='max'),

]
history = model.fit(

        training_set,

        epochs=300,

        validation_data=val_set,

        callbacks = callbacks_list

    )
training_set.class_indices

epochs = len(history.history['loss'])

train_loss = history.history['loss']

val_loss = history.history['val_loss']

train_acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

xc = range(epochs)

print(xc)

print(val_loss)

plt.figure(1,figsize=(7,5))

plt.plot(xc,train_loss)

plt.plot(xc,val_loss)

plt.xlabel('num of Epochs')

plt.ylabel('loss')

plt.title('train_loss vs val_loss')

plt.grid(True)

plt.legend(['train','val'])

#print plt.style.available # use bmh, classic,ggplot for big pictures

plt.style.use(['classic'])



plt.figure(2,figsize=(7,5))

plt.plot(xc,train_acc)

plt.plot(xc,val_acc)

plt.xlabel('num of Epochs')

plt.ylabel('accuracy')

plt.title('train_acc vs val_acc')

plt.grid(True)

plt.legend(['train','val'],loc=4)

#print plt.style.available # use bmh, classic,ggplot for big pictures

plt.style.use(['classic'])
results=model.evaluate(test_set, batch_size=32)

print("test loss, test acc:", results)
import shutil

shutil.rmtree("/kaggle/working/data")
!date