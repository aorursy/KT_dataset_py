import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split



import tensorflow as tf



from keras.models import Sequential, Model, Input

from keras.layers import Conv2D, GlobalAveragePooling2D, Dropout, Dense



from keras.applications import resnet50, mobilenet

from keras.applications.mobilenet import preprocess_input



from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img



from keras.utils import to_categorical



import cv2
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
classes = ["cats", "dogs"]



EPOCHS                  = 10

IMGSIZE                 = 244

BATCH_SIZE              = 32

STOPPING_PATIENCE       = 5

VERBOSE                 = 1

MODEL_NAME              = 'cnn_10epochs_imgsize244'

OPTIMIZER               = 'adam'

TRAINING_DIR            = '../input/cat-and-dog/training_set/training_set'

TEST_DIR                = '../input/cat-and-dog/test_set/test_set'
"""train_datagen = ImageDataGenerator(rescale = 1./255, 

                                   shear_range = 0.2, 

                                   zoom_range = 0.2, 

                                   horizontal_flip = True)"""



train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input) #included in our dependencies



test_datagen = ImageDataGenerator(rescale = 1./255)



training_set = train_datagen.flow_from_directory(TRAINING_DIR, 

                                                 target_size = (IMGSIZE, IMGSIZE), 

                                                 classes=classes,

                                                 batch_size = BATCH_SIZE, 

                                                 class_mode = 'binary')



test_set = test_datagen.flow_from_directory(TEST_DIR, 

                                            target_size = (IMGSIZE, IMGSIZE), 

                                            batch_size = BATCH_SIZE, 

                                            class_mode = 'binary')
j = np.random.randint(32)

X, y = training_set[j]



fig, ax = plt.subplots(5, 6, figsize=(20, 18))



for i, axis in enumerate(ax.flat):

    axis.imshow(X[i])

    cat = classes[int(y[i])]

    axis.set(title=cat)
mobilenet_model = mobilenet.MobileNet(input_shape = (IMGSIZE, IMGSIZE, 3), weights='imagenet', include_top=False)

resnet_model = resnet50.ResNet50(input_shape = (IMGSIZE, IMGSIZE, 3), weights='imagenet', include_top=False)
#inputs  = mobilenet_model.input



inputs = Input(shape=(IMGSIZE, IMGSIZE, 3))

x = mobilenet_model(inputs)

x = GlobalAveragePooling2D()(x)

x = Dense(1024, activation='relu')(x) # we add dense layers so that the model can learn more complex functions and classify for better results.

x = Dropout(0.5)(x) # to avoid overfitting

x = Dense(512, activation='relu')(x) # dense layer 2



outputs = Dense(1, activation='sigmoid')(x)



model = Model(inputs=inputs, outputs=outputs)



model.layers[1].trainable = False



model.summary()
step_size_train = training_set.n//training_set.batch_size

step_size_test = test_set.n//test_set.batch_size



model.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics= ['accuracy'])



history = model.fit_generator(training_set, steps_per_epoch=step_size_train, epochs=EPOCHS, 

                              validation_data = test_set, validation_steps=step_size_test)
model.save(MODEL_NAME + ".h5")

model.save_weight(MODEL_NAME + "_weight.h5")
# summarize history for accuracy

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()