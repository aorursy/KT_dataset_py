import os
os.environ['KAGGLE_USERNAME'] = "lorddexter"
os.environ['KAGGLE_KEY'] = "580247989ab96f8430d502ca9b65afe2"
!kaggle datasets download -d chetankv/dogs-cats-images

!unzip dogs-cats-images.zip

import tensorflow as tf

model = tf.keras.models.Sequential([
                                    tf.keras.layers.Conv2D(filters= 96, kernel_size= (11,11), strides=(4,4), activation='relu', input_shape = (224,224,3), kernel_initializer= tf.keras.initializers.RandomNormal(mean=0, stddev=0.01), bias_initializer=tf.keras.initializers.zeros()),
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.MaxPool2D(pool_size= (3,3), strides= (2,2)),
                                    tf.keras.layers.Conv2D(filters=256, kernel_size=(5,5), activation='relu', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0,stddev=0.01), bias_initializer=tf.keras.initializers.ones()),
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.MaxPool2D(pool_size = (3,3), strides= (2,2)),
                                    tf.keras.layers.Conv2D(filters=384, kernel_size=(3,3), activation='relu', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0,stddev=0.01), bias_initializer=tf.keras.initializers.zeros()),
                                    tf.keras.layers.Conv2D(filters=384, kernel_size=(3,3), activation='relu', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0,stddev=0.01), bias_initializer=tf.keras.initializers.ones()),
                                    tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), activation='relu', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0,stddev=0.01), bias_initializer=tf.keras.initializers.ones()),
                                    tf.keras.layers.MaxPool2D(pool_size= (3,3), strides= (2,2)),
                                    tf.keras.layers.Dense(units=4096, activation='relu', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0,stddev=0.01), bias_initializer=tf.keras.initializers.ones()),
                                    tf.keras.layers.Dropout(rate=0.5),
                                    tf.keras.layers.Dense(units=4096, activation='relu', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0,stddev=0.01), bias_initializer=tf.keras.initializers.ones()),
                                    tf.keras.layers.Dropout(rate=0.5),
                                    tf.keras.layers.Dense(units=4096, activation='relu', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0,stddev=0.01), bias_initializer=tf.keras.initializers.ones()),
                                    tf.keras.layers.Dense(units=2, activation='sigmoid') 
])

model.compile(optimizer= tf.keras.optimizers.SGD(learning_rate= 0.01, momentum=0.9, decay= 0.0005), loss= 'sparse_categorical_crossentropy', metrics= ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
training_data = ImageDataGenerator()
traindata = training_data.flow_from_directory(directory='dataset/training_set', target_size=(224,224), batch_size=128, class_mode='sparse')
test_data = ImageDataGenerator()
testdata = test_data.flow_from_directory(directory='dataset/test_set', target_size=(224,224), batch_size=128, class_mode='sparse')

model.summary()

hist = model.fit(x=traindata, epochs=10, steps_per_epoch=20)