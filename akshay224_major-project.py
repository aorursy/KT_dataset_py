import os

os.listdir("../input/majorprojectdataset")
# Importing Keras libraries and packages

from keras.models import Sequential

from keras.layers import Convolution2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense

from keras.layers import Dropout

from keras.layers.normalization import BatchNormalization



# Initializing the CNN

classifier = Sequential()



# Convolution Step 1

classifier.add(Convolution2D(96, 11, strides = (4, 4), padding = 'valid', input_shape=(224, 224, 3), activation = 'relu'))



# Max Pooling Step 1

classifier.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'valid'))

classifier.add(BatchNormalization())



# Convolution Step 2

classifier.add(Convolution2D(256, 11, strides = (1, 1), padding='valid', activation = 'relu'))



# Max Pooling Step 2

classifier.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding='valid'))

classifier.add(BatchNormalization())



# Convolution Step 3

classifier.add(Convolution2D(384, 3, strides = (1, 1), padding='valid', activation = 'relu'))

classifier.add(BatchNormalization())



# Convolution Step 4

classifier.add(Convolution2D(384, 3, strides = (1, 1), padding='valid', activation = 'relu'))

classifier.add(BatchNormalization())



# Convolution Step 5

classifier.add(Convolution2D(256, 3, strides=(1,1), padding='valid', activation = 'relu'))



# Max Pooling Step 3

classifier.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'valid'))

classifier.add(BatchNormalization())



# Flattening Step

classifier.add(Flatten())



# Full Connection Step

classifier.add(Dense(units = 4096, activation = 'relu'))

classifier.add(Dropout(0.4))

classifier.add(BatchNormalization())

classifier.add(Dense(units = 4096, activation = 'relu'))

classifier.add(Dropout(0.4))

classifier.add(BatchNormalization())

classifier.add(Dense(units = 1000, activation = 'relu'))

classifier.add(Dropout(0.2))

classifier.add(BatchNormalization())

classifier.add(Dense(units = 39, activation = 'softmax'))

classifier.summary()
# let's visualize layer names and layer indices to see how many layers

# we should freeze:

from keras import layers

for i, layer in enumerate(classifier.layers):

   print(i, layer.name)
#trainable parameters decrease after freezing some bottom layers   

classifier.summary()
# Compiling the Model

from keras import optimizers

classifier.compile(optimizer=optimizers.SGD(lr=0.001, momentum=0.9, decay=0.005),

              loss='categorical_crossentropy',

              metrics=['accuracy'])
# image preprocessing

from keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(rescale=1./255,

                                   shear_range=0.2,

                                   zoom_range=0.2,

                                   width_shift_range=0.2,

                                   height_shift_range=0.2,

                                   fill_mode='nearest')



valid_datagen = ImageDataGenerator(rescale=1./255)



batch_size = 128

base_dir = "../input/majorprojectdataset/plant-diseases-classification-using-alexnet/plant-diseases-classification-using-alexnet"



training_set = train_datagen.flow_from_directory(base_dir+'/train',

                                                 target_size=(224, 224),

                                                 batch_size=batch_size,

                                                 class_mode='categorical')



valid_set = valid_datagen.flow_from_directory(base_dir+'/valid',

                                            target_size=(224, 224),

                                            batch_size=batch_size,

                                            class_mode='categorical')
class_dict = training_set.class_indices

print(class_dict)
li = list(class_dict.keys())

print(li)
train_num = training_set.samples

valid_num = valid_set.samples

print(train_num, valid_num)
# checkpoint

from keras.callbacks import ModelCheckpoint

weightpath = "../input/majorprojectdataset/plant-diseases-classification-using-alexnet/best_weights_9.hdf5"

checkpoint = ModelCheckpoint(weightpath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='max')

callbacks_list = [checkpoint]



#fitting images to CNN

history = classifier.fit_generator(training_set,

                         steps_per_epoch=train_num//batch_size,

                         validation_data=valid_set,

                         epochs=25,

                         validation_steps=valid_num//batch_size,

                         callbacks=callbacks_list)

#saving model

filepath="AlexNetModel.hdf5"

classifier.save(filepath)



export_dir = "h5model"

classifier.save(export_dir)



import tensorflow as tf

# Convert the model.

converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)

tflite_model = converter.convert()



# Save the TF Lite model.

with tf.io.gfile.GFile('model.tflite', 'wb') as f:

  f.write(tflite_model)



# Convert the model.

converter = tf.lite.TFLiteConverter.from_keras_model(classifier)

tflite_model = converter.convert()



# Save the TF Lite model.

with tf.io.gfile.GFile('my_model.tflite', 'wb') as f:

  f.write(tflite_model)

#plotting training values

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)



#accuracy plot

plt.plot(epochs, acc, color='green', label='Training Accuracy')

plt.plot(epochs, val_acc, color='blue', label='Validation Accuracy')

plt.title('Training and Validation Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend()



plt.figure()

#loss plot

plt.plot(epochs, loss, color='pink', label='Training Loss')

plt.plot(epochs, val_loss, color='red', label='Validation Loss')

plt.title('Training and Validation Loss')

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.legend()



plt.show()
print("[INFO] Calculating model accuracy")

scores = classifier.evaluate(valid_set)

print(f"Test Accuracy: {scores[1]*100}")
import keras

classifier = keras.models.load_model('AlexNetModel.hdf5')

from tensorflow import lite

converter = lite.TFLiteConverter.from_keras_model(classifier)

tflite_model = converter.convert()

open('tf_liteModel.tflite', 'wb').write(tflite_model)
