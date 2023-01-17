import tensorflow as tf

import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

from keras.applications.inception_v3 import InceptionV3



print("Setup Complete!")


train_DIR = "/kaggle/input/intel-image-classification/seg_train/seg_train/"



train_datagen = ImageDataGenerator( rescale = 1.0/255,

                                          width_shift_range=0.2,

                                          height_shift_range=0.2,

                                          zoom_range=0.2,

                                          vertical_flip=True,

                                          fill_mode='nearest')





train_generator = train_datagen.flow_from_directory(train_DIR,

                                                    batch_size=32,

                                                    class_mode='categorical',

                                                    target_size=(150, 150))



test_DIR = "/kaggle/input/intel-image-classification/seg_test/seg_test/"

validation_datagen = ImageDataGenerator(rescale = 1.0/255)





validation_generator = validation_datagen.flow_from_directory(test_DIR,

                                                    batch_size=128,

                                                    class_mode='categorical',

                                                    target_size=(150, 150))
inceptionV3 = InceptionV3(include_top= False, input_shape=(150,150,3))



for layer in inceptionV3.layers:

	layer.trainable = False
last_layer = inceptionV3.get_layer('mixed9')



print('last layer output shape: ', last_layer.output_shape)



last_output = last_layer.output


x = tf.keras.layers.Flatten()(last_output)

x = tf.keras.layers.Dense(units = 1024, activation = tf.nn.relu)(x)

x = tf.keras.layers.Dropout(0.2)(x)

x = tf.keras.layers.Dense  (6, activation = tf.nn.softmax)(x)



model = tf.keras.Model( inceptionV3.input, x)



learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',

                                            patience=1,

                                            verbose=1,

                                            factor=0.25,

                                            min_lr=0.000003)



model.compile(loss = 'categorical_crossentropy', optimizer= tf.keras.optimizers.Adam(), metrics=['acc'])



model.summary()
history = model.fit(train_generator,

                    epochs = 10,

                    verbose = 1,

                   validation_data = validation_generator,

                   callbacks=[learning_rate_reduction])
%matplotlib inline

acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(acc))



plt.plot(epochs, acc, 'r', label='Training accuracy')

plt.plot(epochs, val_acc, 'b', label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend()

plt.figure()



plt.plot(epochs, loss, 'r', label='Training Loss')

plt.plot(epochs, val_loss, 'b', label='Validation Loss')

plt.title('Training and validation loss')

plt.legend()



plt.show()