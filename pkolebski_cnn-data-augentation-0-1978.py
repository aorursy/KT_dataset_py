import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt

%matplotlib inline
train = pd.read_json("../input/train.json")
def standardise_data(data):

    '''standardise vector'''

    out = (np.array(data) - np.mean(data)) / np.std(data)

    return out.tolist()



def process_data(data, predict=False):

    data["band_1"] = data["band_1"].apply(standardise_data)

    data["band_2"] = data["band_2"].apply(standardise_data)

    band1 = np.array([np.array(band).astype(np.float32).reshape(75,75) for band in data["band_1"]])

    band2 = np.array([np.array(band).astype(np.float32).reshape(75,75) for band in data["band_2"]])

    X = np.concatenate([band1[:, :, :, np.newaxis], 

                            band2[:, :, :, np.newaxis],

                            ((band1+band2)/2)[:, :, :, np.newaxis]], 

                            axis=-1)

    if predict==False:

        y = np.array(data['is_iceberg'])

        return X, y

    return X
X_train, y_train = process_data(train)
plt.imshow(X_train[80,:,:,0])

print(y_train[80])
plt.imshow(X_train[91,:,:,0])

print(y_train[91])
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, Activation, BatchNormalization

from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
def getModel():

    drop = 0.2



    gmodel=Sequential()



    gmodel.add(Conv2D(32, kernel_size=(3, 3),activation='relu', input_shape=(75, 75, 3)))

    gmodel.add(BatchNormalization())

    gmodel.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    gmodel.add(Dropout(drop))



    gmodel.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))

    gmodel.add(BatchNormalization())

    gmodel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    gmodel.add(Dropout(drop))



    gmodel.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))

    gmodel.add(BatchNormalization())

    gmodel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    gmodel.add(Dropout(drop))



    gmodel.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))

    gmodel.add(BatchNormalization())

    gmodel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    gmodel.add(Dropout(drop))



    gmodel.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))

    gmodel.add(BatchNormalization())

    gmodel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    gmodel.add(Dropout(drop))



    gmodel.add(Flatten())



    gmodel.add(Dense(512))

    gmodel.add(BatchNormalization())

    gmodel.add(Activation('relu'))

    gmodel.add(Dropout(drop))



    gmodel.add(Dense(256))

    gmodel.add(BatchNormalization())

    gmodel.add(Activation('relu'))

    gmodel.add(Dropout(drop))



    gmodel.add(Dense(1))

    gmodel.add(Activation('sigmoid'))



    mypotim=Adam(lr=0.0008, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=True)

    gmodel.compile(loss='binary_crossentropy',

                  optimizer=mypotim,

                  metrics=['accuracy'])

    gmodel.summary()

    return gmodel





def get_callbacks(filepath, patience=2):

    es = EarlyStopping('val_loss', patience=patience, mode="min")

    msave = ModelCheckpoint(filepath, save_best_only=True)

    return [es, msave]

file_path = "model_weights3.hdf5"

callbacks = get_callbacks(filepath=file_path, patience=5)

X_train_cv, X_valid, y_train_cv, y_valid = train_test_split(X_train, y_train, random_state=1, test_size=0.25)
gen = ImageDataGenerator(rotation_range=10, width_shift_range=0.3, height_shift_range=0.3, 

                         zoom_range=0.2, horizontal_flip=True, vertical_flip=True)

gen.fit(X_train_cv)
gmodel=getModel()



history = gmodel.fit_generator(gen.flow(X_train_cv, y_train_cv, batch_size=32),steps_per_epoch=256,

                    epochs=50, verbose=1, validation_data=(X_valid, y_valid), callbacks=callbacks)
gmodel.load_weights(filepath=file_path)

score = gmodel.evaluate(X_valid, y_valid, verbose=1)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
plt.plot(history.epoch, history.history['acc'], 'b',history.epoch, history.history['val_acc'], 'g')

plt.show()

plt.plot(history.epoch, history.history['loss'], 'b',history.epoch, history.history['val_loss'], 'g')

plt.show()
pred = np.round(gmodel.predict_proba(X_valid[0:3,:,:,:]))

for i in range(3):

    print('real:', y_valid[i], 'predicted:', pred[i])

    plt.imshow(X_valid[i,:,:,0])

    plt.show()

gmodel=getModel()

gmodel.load_weights(filepath=file_path)



test = pd.read_json("../input/test.json")

X_test = process_data(test, predict=True)



predicted_test=gmodel.predict_proba(X_test)



submission = pd.DataFrame()

submission['id'] = test['id']

submission['is_iceberg'] = predicted_test.reshape((predicted_test.shape[0]))

submission.to_csv('sub3.csv', index=False)



del test

del X_test