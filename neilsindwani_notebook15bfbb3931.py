#importing libraries
import numpy as np 
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.layers import Dense, Dropout, Conv2D, BatchNormalization, Flatten, Activation, MaxPool2D, SpatialDropout2D
import os
# Data preprosseccing
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   width_shift_range=0.1,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('../input/intel-image-classification/seg_train/seg_train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,             
                                                 class_mode = 'categorical')
test_set = train_datagen.flow_from_directory('../input/intel-image-classification/seg_test/seg_test',
                                                 target_size = (64, 64))
# Neural Network 
import tensorflow.keras.optimizers as Optimizer
model3 = Sequential()
val_ReduceLROnPlateau2 = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',
                                                             patience=3,
                                                             verbose=1,
                                                             factor=0.1,
                                                             min_lr=1e-8)
callback2 = tf.keras.callbacks.ModelCheckpoint(
    filepath="model2.hdf5",
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)
model3.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
model3.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
model3.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
model3.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
model3.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
model3.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
model3.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
model3.add(tf.keras.layers.Flatten())
model3.add(tf.keras.layers.Dense(units=128, activation='relu'))
model3.add(Dropout(0.5))
model3.add(tf.keras.layers.Dense(units=6, activation='softmax'))
model3.compile(optimizer="adam",loss='categorical_crossentropy',metrics=['accuracy'])
#model3.summary()
history3 = model3.fit(x = training_set, validation_data = test_set, epochs = 25, callbacks = [val_ReduceLROnPlateau2, callback2], verbose = 1)#Callbacks, epochs etc.
indi = training_set.class_indices
print(indi)
correct = []
for i in sorted(os.listdir('../input/intel-image-classification/seg_test/seg_test')):
    for j in sorted(os.listdir('../input/intel-image-classification/seg_test/seg_test/'+i)):
        correct.append(indi[i])
# Identifying the correct images
model3.load_weights("model2.hdf5") 
results = model3.evaluate_generator(test_set)
print("Accuracy:",results[1])
plt.plot(history3.history['loss'], label='train loss')
plt.plot(history3.history['val_loss'], label='test loss')
plt.ylabel('loss')
plt.xlabel('No. of epochs')
plt.legend(loc="upper left")
plt.show()
plt.plot(history3.history['accuracy'], label='train accuracy')
plt.plot(history3.history['val_accuracy'], label='test accuracy')
plt.ylabel('accuracy')
plt.xlabel('No. of epochs')
plt.legend(loc="upper left")
plt.show()
plt.imshow(mpimg.imread('../input/intel-image-classification/seg_test/seg_test/buildings/'+os.listdir('../input/intel-image-classification/seg_test/seg_test/buildings')[0]))
plt.title("Correct: "+"0 "+"Pred: "+str(np.argmax(results[0])))