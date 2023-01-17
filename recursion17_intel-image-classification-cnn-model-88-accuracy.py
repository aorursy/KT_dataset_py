import tensorflow as tf

import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau



print("Setup Complete!")
train_DIR = "/kaggle/input/intel-image-classification/seg_train/seg_train/"



train_datagen = ImageDataGenerator( rescale = 1.0/255,

                                          width_shift_range=0.2,

                                          height_shift_range=0.2,

                                          zoom_range=0.2,

                                          horizontal_flip=True,

                                          fill_mode='nearest')





train_generator = train_datagen.flow_from_directory(train_DIR,

                                                    batch_size=32,

                                                    class_mode='categorical',

                                                    target_size=(150, 150))



test_DIR = "/kaggle/input/intel-image-classification/seg_test/seg_test/"

validation_datagen = ImageDataGenerator(rescale = 1.0/255)





validation_generator = validation_datagen.flow_from_directory(test_DIR,

                                                    batch_size=64,

                                                    class_mode='categorical',

                                                    target_size=(150, 150))


model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(8, (3,3), activation=tf.nn.relu,input_shape=(150, 150, 3)),

    tf.keras.layers.MaxPooling2D(2,2),

    

    tf.keras.layers.Conv2D(16, (3,3), activation=tf.nn.relu),

    tf.keras.layers.MaxPooling2D(2,2),



    tf.keras.layers.Conv2D(32, (3,3), activation=tf.nn.relu),

    tf.keras.layers.MaxPooling2D(2,2),

    

    tf.keras.layers.Conv2D(64, (3,3), activation=tf.nn.relu,padding = 'Same'),

    tf.keras.layers.MaxPooling2D(2,2),

    

    

    #tf.keras.layers.Conv2D(256, (3,3), activation=tf.nn.relu,padding = 'Same'),

    #tf.keras.layers.MaxPooling2D(2,2),

    

    tf.keras.layers.Flatten(),



    tf.keras.layers.Dense(512, activation=tf.nn.relu),

    #tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(6, activation = tf.nn.softmax)

])



model.summary()
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',

                                            patience=2,

                                            verbose=1,

                                            factor=0.5,

                                            min_lr=0.000003)
model.compile(loss = 'categorical_crossentropy', optimizer= tf.keras.optimizers.Adam(), metrics=['acc'])



history = model.fit(train_generator,

                    epochs = 25,

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


