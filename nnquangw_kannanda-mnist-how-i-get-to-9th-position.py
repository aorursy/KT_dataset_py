import numpy as np

import pandas as pd

import tensorflow as tf

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder 

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
# Data path

train_set = "../input/Kannada-MNIST/train.csv"

valid_set = "../input/Kannada-MNIST/Dig-MNIST.csv"

test_set = "../input/Kannada-MNIST/test.csv"



# Read data

train_csv = pd.read_csv(train_set)

valid_csv = pd.read_csv(valid_set)

test_csv = pd.read_csv(test_set)
def image_generator(images_csv, lbl=True):

    labels = 0

    if (lbl==True):

        labels = images_csv[images_csv.columns[0]].to_numpy(dtype=np.float64, copy=True)



        labels = tf.keras.utils.to_categorical(labels, num_classes=10)



    images = images_csv.loc[:,'pixel0':'pixel783'].to_numpy(dtype=np.float64, copy=True)

    # Reshape 28x28x1

    images = images.reshape((len(images),28,28,1))

    # Normalization

    images = images / 255.

    return images, labels
train_images, train_labels = image_generator(train_csv)

valid_images, valid_labels = image_generator(valid_csv)
X_train, y_train, X_test, y_test = train_test_split(np.concatenate((train_images, valid_images)),

                                                   np.concatenate((train_labels, valid_labels)),

                                                   test_size=0.1,

                                                   shuffle=True)
def cnn_model():

    inp = tf.keras.Input(shape=(28,28,1))

    x1 = tf.keras.layers.Conv2D(128, (1,1), strides=(1,1), activation='relu')(inp)

    x1 = tf.keras.layers.BatchNormalization()(x1)

    x3 = tf.keras.layers.Conv2D(128, (3,3), padding='same', strides=(1,1), activation='relu')(inp)

    x3 = tf.keras.layers.BatchNormalization()(x3)

    x5 = tf.keras.layers.Conv2D(128, (5,5), padding='same', strides=(1,1), activation='relu')(inp)

    x5 = tf.keras.layers.BatchNormalization()(x5)



    averaged = tf.keras.layers.Average()([x1,x3,x5])

    averaged = tf.keras.layers.Activation('relu')(averaged)

    averaged = tf.keras.layers.BatchNormalization()(averaged)



    x = tf.keras.layers.Conv2D(128, (5,5), strides=(1,1), activation='relu')(averaged)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Dropout(rate=0.25)(x)





    x1 = tf.keras.layers.Conv2D(128, (1,1), strides=(1,1), activation='relu')(x)

    x1 = tf.keras.layers.BatchNormalization()(x1)

    x3 = tf.keras.layers.Conv2D(128, (3,3), padding='same', strides=(1,1), activation='relu')(x)

    x3 = tf.keras.layers.BatchNormalization()(x3)

    x5 = tf.keras.layers.Conv2D(128, (5,5), padding='same', strides=(1,1), activation='relu')(x)

    x5 = tf.keras.layers.BatchNormalization()(x5)



    averaged = tf.keras.layers.Average()([x1,x3,x5])

    averaged = tf.keras.layers.Activation('relu')(averaged)

    averaged = tf.keras.layers.BatchNormalization()(averaged)



    x = tf.keras.layers.Conv2D(128, (3,3), strides=(1,1), activation='relu')(averaged)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Dropout(rate=0.25)(x)



    x1 = tf.keras.layers.Conv2D(256, (1,1), strides=(1,1), activation='relu')(x)

    x1 = tf.keras.layers.BatchNormalization()(x1)

    x3 = tf.keras.layers.Conv2D(256, (3,3), padding='same', strides=(1,1), activation='relu')(x)

    x3 = tf.keras.layers.BatchNormalization()(x3)

    x5 = tf.keras.layers.Conv2D(256, (5,5), padding='same', strides=(1,1), activation='relu')(x)

    x5 = tf.keras.layers.BatchNormalization()(x5)



    averaged = tf.keras.layers.Average()([x1,x3,x5])

    averaged = tf.keras.layers.Activation('relu')(averaged)

    averaged = tf.keras.layers.BatchNormalization()(averaged)



    x = tf.keras.layers.Conv2D(256, (1,1), strides=(1,1), activation='relu')(averaged)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Dropout(rate=0.25)(x)



    x1 = tf.keras.layers.Conv2D(256, (1,1), strides=(1,1), activation='relu')(x)

    x1 = tf.keras.layers.BatchNormalization()(x1)

    x3 = tf.keras.layers.Conv2D(256, (3,3), padding='same', strides=(1,1), activation='relu')(x)

    x3 = tf.keras.layers.BatchNormalization()(x3)

    x5 = tf.keras.layers.Conv2D(256, (5,5), padding='same', strides=(1,1), activation='relu')(x)

    x5 = tf.keras.layers.BatchNormalization()(x5)



    averaged = tf.keras.layers.Average()([x1,x3,x5])

    averaged = tf.keras.layers.Activation('relu')(averaged)

    averaged = tf.keras.layers.BatchNormalization()(averaged)



    x = tf.keras.layers.Conv2D(256, (1,1), strides=(1,1), activation='relu')(averaged)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Dropout(rate=0.25)(x)

    

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(256, activation=tf.nn.relu)(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dense(128, activation=tf.nn.relu)(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dense(64, activation=tf.nn.relu)(x)

    x = tf.keras.layers.BatchNormalization()(x)



    

    output = tf.keras.layers.Dense(10, activation=tf.nn.softmax)(x)



    model = tf.keras.Model(inputs=inp, outputs=output)

    return model
model = cnn_model()

model.summary()
model.load_weights("../input/pretrained/model2.h5")
EPOCHS = 1

BATCH_SIZE = 128

lr = 0.00000001

opt = tf.keras.optimizers.Adam(learning_rate=lr)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', 

                                            patience=10,

                                            verbose=1,

                                            factor=0.75)

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)


train_aug = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=10,

                                                  width_shift_range=0.2,

                                                  height_shift_range=0.2,

                                                  shear_range=0.1,

                                                  zoom_range=0.2,

                                                  horizontal_flip=False)

valid_aug = tf.keras.preprocessing.image.ImageDataGenerator()



mcp = tf.keras.callbacks.ModelCheckpoint("../input/pretrained-models/model.h5", monitor='val_loss',

                                             save_best_only=True, save_weights_only=True)

model.compile(optimizer=opt, loss=tf.keras.losses.CategoricalCrossentropy(),metrics=['accuracy'])

model.fit_generator(train_aug.flow(train_images, train_labels,batch_size=BATCH_SIZE),

                    steps_per_epoch=10,

                    validation_data=valid_aug.flow(valid_images, valid_labels),

                    validation_steps=50,

                    epochs=EPOCHS, verbose=1,

                       callbacks=[learning_rate_reduction, es])
test_images, _ = image_generator(test_csv, lbl=False)



preds = model.predict(test_images)

preds = preds.argmax(axis=1)

print(preds)

# predictions to dataframe

preds = preds.astype(int).flatten()

preds = (LabelEncoder().fit_transform((preds)))

preds = pd.DataFrame({'label': preds})
sub = pd.DataFrame(data=test_csv.id)

sub = sub.join(preds)

# Write out the predictions to disk

sub.to_csv('submission.csv', index=False)