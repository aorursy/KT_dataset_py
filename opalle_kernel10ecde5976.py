import numpy as np

import pandas as pd

from keras import layers, optimizers

from keras.models import Model

from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

import tensorflow as tf



plt.style.use('ggplot')

%matplotlib inline
test = pd.read_csv('../input/Kannada-MNIST/test.csv')

train = pd.read_csv('../input/Kannada-MNIST/train.csv')

dig_df = pd.read_csv('../input/Kannada-MNIST/Dig-MNIST.csv')

sample_df = pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')
# convert dataframes to numpy matricies

X = train.drop('label', axis=1).to_numpy()

y = train['label'].to_numpy()

X_test = test.drop('id', axis=1).to_numpy()

X_dig = dig_df.drop('label', axis=1).to_numpy()

y_dig = dig_df['label'].to_numpy()



# reshape X's for keras and encode y using one-hot-vector-encoding

X = X.reshape(-1, 28, 28, 1)

y = to_categorical(y)

X_test = X_test.reshape(-1, 28, 28, 1)

X_dig = X_dig.reshape(-1, 28, 28, 1)



# normalize the data to range(0, 1)

X = X / 255

X_dig = X_dig / 255

X_test = X_test / 255

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.15, random_state=42)
train_datagen = ImageDataGenerator(rescale=1.0,

                                   rotation_range=10,

                                   width_shift_range=0.25,

                                   height_shift_range=0.25,

                                   shear_range=0.1,

                                   zoom_range=0.25,

                                   horizontal_flip=False)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 

                                            patience=7, 

                                            verbose=1, 

                                            factor=0.1, 

                                            min_lr=1e-8)
def build_norm_model():

      inputs = layers.Input(shape=(28, 28, 1))



      x = layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', input_shape=(28, 28, 1))(inputs)

      x = layers.LeakyReLU(alpha=0.3)(x)

      x = layers.BatchNormalization()(x)  

      x = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1))(x)

      x = layers.LeakyReLU(alpha=0.3)(x)

      x = layers.BatchNormalization()(x)

      x = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1))(x)

      x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

      x = layers.LeakyReLU(alpha=0.3)(x)

      x = layers.BatchNormalization()(x)

      x = layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1))(x)

      x = layers.LeakyReLU(alpha=0.3)(x)

      x = layers.BatchNormalization()(x)

      x = layers.Dropout(0.5)(x)

      x = layers.BatchNormalization()(x)

      x = layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1))(x)

      x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

      x = layers.LeakyReLU(alpha=0.3)(x)

      x = layers.BatchNormalization()(x)

      x = layers.Flatten()(x)

      x = layers.Dense(units=256)(x)

      x = layers.LeakyReLU(alpha=0.3)(x)

      x = layers.BatchNormalization()(x)

      x = layers.Dropout(0.5)(x)

      output = layers.Dense(10, activation='softmax')(x)



      return Model(inputs=inputs, outputs=output)
def plot_val_accuracy(histories, y_label, epochs):

  plt.figure(figsize=(15, 8))

  epoch_range = np.arange(epochs - 10)

  for id, history in enumerate(histories):

    plt.plot(epoch_range, history.history['val_accuracy'][10:], label='batch size=' + str(y_label[id]))

    plt.plot(epoch_range, history.history['accuracy'][10:], label='batch size=' + str(y_label[id]))

    plt.title('Model validation accuracy')

    plt.ylabel('accuracy')

    plt.xlabel('epoch')

    plt.legend()
def make_prediction(model, x):

    y_pred = model.predict(x)

    return np.argmax(y_pred, axis=1)
#y_add = make_prediction(sgd_norm_model, X_dig)



#idx = y_add == y_dig

#X_train = np.concatenate((X_train, X_dig[idx]))

#y_dig = to_categorical(y_dig)

#y_train = np.concatenate((y_train, y_dig))

#X_train = np.concatenate((X_train, X_dig))

#y_train = np.concatenate((y_train, y_dig[idx]))



#y_test_add = make_prediction(sgd_norm_model, X_test)

#X_train = np.concatenate((X_train, X_test))

#y_train = np.concatenate((y_train,to_categorical(y_test_add)))



#shuffler = np.random.permutation(X_train.shape[0])

#X_train = X_train[shuffler]

#y_train = y_train[shuffler]
final_model = build_norm_model()

final_model.compile(optimizer=optimizers.RMSprop(lr=0.001),

              loss='categorical_crossentropy',

              metrics=['accuracy'])

history_final = final_model.fit_generator(train_datagen.flow(X_train, y_train, batch_size=1024),

                                           steps_per_epoch=100,

                                           epochs=120,

                                           validation_data=(X_valid, y_valid),

                                           callbacks=[learning_rate_reduction],

                                           verbose=1)
plot_val_accuracy([history_final], [1024], len(history_final.history['val_accuracy']))
y_result = make_prediction(final_model, X_test)



# save predictions

sample_df['label'] = y_result

sample_df.to_csv('submission.csv',index=False)