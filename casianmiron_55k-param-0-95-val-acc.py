import tensorflow as tf
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(50, 50, 3)),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation='relu'),

    tf.keras.layers.Dense(2, activation='softmax')

])
model.summary()
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'Adam', metrics = ['accuracy'])

from tensorflow.keras.preprocessing.image import ImageDataGenerator



# All images will be rescaled by 1./255

train_datagen = ImageDataGenerator(rescale=1./255,validation_split=0.2)



# Flow training images in batches of 128 using train_datagen generator

train_generator = train_datagen.flow_from_directory(

        "../input/cell_images/cell_images/",  # This is the source directory for training images

        target_size=(50, 50),  # All images will be resized to 50x50

        batch_size=128,

        

        class_mode='binary',

        subset='training'

        )



validation_generator=train_datagen.flow_from_directory( "../input/cell_images/cell_images/",

                                               target_size=(50,50),

                                               batch_size=32,

                                               class_mode='binary',

                                               subset='validation'

                                               )
history = model.fit_generator(

      train_generator,

      steps_per_epoch=8,  

      epochs=30,

      verbose=2,

      validation_data=validation_generator,

      validation_steps=8)
