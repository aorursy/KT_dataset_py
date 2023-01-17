import numpy as np

import pandas as pd

import tensorflow as tf

from tensorflow.keras.models import load_model

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
# Data path

train_set = "../input/Kannada-MNIST/train.csv"

valid_set = "../input/Kannada-MNIST/Dig-MNIST.csv"

test_set = "../input/Kannada-MNIST/test.csv"



# Read data

train = pd.read_csv(train_set)

valid = pd.read_csv(valid_set)

test = pd.read_csv(test_set)



xtrain = train.drop('label', axis=1)

ytrain = train.label

xtest = test.drop('id', axis=1)

xval = valid.drop('label', axis=1)

yval = valid.label



xtrain = xtrain.values

ytrain = ytrain.values

xval = xval.values

yval = yval.values

xtest = xtest.values



xtrain = xtrain.reshape(-1, 28, 28, 1).astype('float32')

xval = xval.reshape(-1, 28, 28, 1).astype('float32')

xtest = xtest.reshape(-1, 28, 28, 1).astype('float32')



xtrain = (xtrain - np.mean(xtrain))/255

xval = (xval - np.mean(xval))/255

xtest = (xtest - np.mean(xtest))/255
def build_model():

    inputs = tf.keras.Input(shape=(28,28,1))

    

    x1 = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(inputs)

    x1 = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform")(x1)

    x1 = tf.keras.layers.LeakyReLU(alpha=0.1)(x1)



    x2 = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(x1)

    x2 = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform")(x2)

    x2 = tf.keras.layers.LeakyReLU(alpha=0.1)(x2)



    x3 = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(x2)

    x3 = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform")(x3)

    x3 = tf.keras.layers.LeakyReLU(alpha=0.1)(x3)



#     x4 = tf.keras.layers.concatenate([x1, x3], axis=-1)

#     x4 = tf.keras.layers.Add()([inputs, x3])

    x4 = tf.keras.layers.Average()([x1, x2, x3])

    x4 = tf.keras.layers.MaxPooling2D((2, 2))(x4)

    x4 = tf.keras.layers.Dropout(0.25)(x4)



    x5 = tf.keras.layers.Conv2D(128, (3, 3), padding='same')(x4)

    x5 = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform")(x5)

    x5 = tf.keras.layers.LeakyReLU(alpha=0.1)(x5)



    x6 = tf.keras.layers.Conv2D(128, (3, 3), padding='same')(x5)

    x6 = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform")(x6)

    x6 = tf.keras.layers.LeakyReLU(alpha=0.1)(x6)



    x7 = tf.keras.layers.Conv2D(128, (3, 3), padding='same')(x6)

    x7 = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform")(x7)

    x7 = tf.keras.layers.LeakyReLU(alpha=0.1)(x7)



#     x8 = tf.keras.layers.concatenate([x5, x7], axis=-1)

#     x8 = tf.keras.layers.Add()([x5, x7])

    x8 = tf.keras.layers.Average()([x5, x6, x7])

    x8 = tf.keras.layers.MaxPooling2D((2, 2))(x8)

    x8 = tf.keras.layers.Dropout(0.25)(x8)

 

    x9 = tf.keras.layers.Conv2D(256, (3, 3), padding='same')(x8)

    x9 = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform")(x9)

    x9 = tf.keras.layers.LeakyReLU(alpha=0.1)(x9)



    x10 = tf.keras.layers.Conv2D(256, (3, 3), padding='same')(x9)

    x10 = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform")(x10)

    x10 = tf.keras.layers.LeakyReLU(alpha=0.1)(x10)



    x11 = tf.keras.layers.Conv2D(256, (3, 3), padding='same')(x10)

    x11 = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform")(x11)

    x11 = tf.keras.layers.LeakyReLU(alpha=0.1)(x11)



    x12 = tf.keras.layers.Average()([x9, x10, x11])

    x12 = tf.keras.layers.MaxPooling2D((2, 2))(x12)

    x12 = tf.keras.layers.Dropout(0.25)(x12)

    



    x = tf.keras.layers.Flatten()(x12)

    x = tf.keras.layers.Dense(256, activation=tf.nn.relu)(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dense(128, activation=tf.nn.relu)(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dense(64, activation=tf.nn.relu)(x)

    x = tf.keras.layers.BatchNormalization()(x)

    

    output = tf.keras.layers.Dense(10, activation=tf.nn.softmax)(x)



    model = tf.keras.Model(inputs=inputs, outputs=output)

    model.summary()

    return model
opt = tf.keras.optimizers.RMSprop(learning_rate=0.002, rho=0.9)

model = build_model()

model.compile(optimizer=opt,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
data_generator = tf.keras.preprocessing.image.ImageDataGenerator(

                                                rotation_range=12,

                                                zoom_range=0.35,

                                                width_shift_range=0.3,

                                                height_shift_range=0.3)



data_generator.fit(xtrain)



learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(

                                                monitor='loss',

                                                factor=0.2,

                                                patience=2,

                                                verbose=2,

                                                mode="min",

                                                min_delta=0.0001,

                                                cooldown=0,

                                                min_lr=0.0000001)



es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=300, restore_best_weights=True)
model.fit_generator(data_generator.flow(xtrain, ytrain, batch_size=1024), 

                                       steps_per_epoch=len(xtrain)//1024,

                                       epochs=500,

                                       validation_data=(np.array(xval), np.array(yval)),

                                       validation_steps=50,

                                       callbacks=[learning_rate_reduction, es])
preds = model.predict(xtest)

preds = preds.argmax(axis=1)

print(preds)

# predictions to dataframe

preds = preds.astype(int).flatten()

preds = (LabelEncoder().fit_transform((preds)))

preds = pd.DataFrame({'label': preds})



sub = pd.DataFrame(data=test.id)

sub = sub.join(preds)



sub.to_csv('submission.csv', index=False)