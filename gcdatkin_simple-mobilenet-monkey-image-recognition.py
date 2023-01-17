import numpy as np



import tensorflow as tf
train_dir = '../input/10-monkey-species/training/training'

test_dir = '../input/10-monkey-species/validation/validation'
IMAGE_HEIGHT = 224

IMAGE_WIDTH = 224



BATCH_SIZE = 16
train_generator = tf.keras.preprocessing.image.ImageDataGenerator(

    rescale=1./255,

    rotation_range=50,

    width_shift_range=0.5,

    height_shift_range=0.5,

    validation_split=0.2

)



test_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
train_data = train_generator.flow_from_directory(

    train_dir,

    target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),

    batch_size=BATCH_SIZE,

    subset='training'

)



validation_data = train_generator.flow_from_directory(

    train_dir,

    target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),

    batch_size=BATCH_SIZE,

    subset='validation'

)



test_data = test_generator.flow_from_directory(

    test_dir,

    target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),

    batch_size=BATCH_SIZE

)
pretrained_model = tf.keras.applications.MobileNet(

    weights='imagenet',

    include_top=False,

    pooling='avg',

    input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)

)



pretrained_model.trainable = False
inputs = tf.keras.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3))

x = pretrained_model(inputs, training=False)

x = tf.keras.layers.Dense(1024, activation='relu')(x)

x = tf.keras.layers.Dense(1024, activation='relu')(x)

x = tf.keras.layers.Dense(512, activation='relu')(x)

outputs = tf.keras.layers.Dense(10, activation='softmax')(x)



model = tf.keras.Model(inputs, outputs)
model.summary()
model.compile(

    optimizer='adam',

    loss='categorical_crossentropy',

    metrics=['accuracy']

)





EPOCHS = 30



history = model.fit(

    train_data,

    validation_data=validation_data,

    batch_size=BATCH_SIZE,

    epochs=EPOCHS,

    callbacks=[

        tf.keras.callbacks.EarlyStopping(

            monitor='val_loss',

            patience=3,

            restore_best_weights=True,

            verbose=1

        )

    ],

    verbose=2

)
model.evaluate(test_data)