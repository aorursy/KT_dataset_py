import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from keras.applications.vgg16 import VGG16

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from keras.layers import Input, Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.models import Sequential, Model

from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator

np.random.seed(7)
def make_df(path, mode):

    """

    params

    --------

    path(str): path to json

    mode(str): "train" or "test"



    outputs

    --------

    X(np.array): list of images shape=(None, 75, 75, 3)

    Y(np.array): list of labels shape=(None,)

    df(pd.DataFrame): data frame from json

    """

    df = pd.read_json(path)

    df.inc_angle = df.inc_angle.replace('na', 0)

    X = _get_scaled_imgs(df)

    if mode == "test":

        return X, df



    Y = np.array(df['is_iceberg'])



    idx_tr = np.where(df.inc_angle > 0)



    X = X[idx_tr[0]]

    Y = Y[idx_tr[0], ...]



    return X, Y





def _get_scaled_imgs(df):

    imgs = []



    for i, row in df.iterrows():

        band_1 = np.array(row['band_1']).reshape(75, 75)

        band_2 = np.array(row['band_2']).reshape(75, 75)

        band_3 = band_1 + band_2



        a = (band_1 - band_1.mean()) / (band_1.max() - band_1.min())

        b = (band_2 - band_2.mean()) / (band_2.max() - band_2.min())

        c = (band_3 - band_3.mean()) / (band_3.max() - band_3.min())



        imgs.append(np.dstack((a, b, c)))



    return np.array(imgs)
def SmallCNN():

    model = Sequential()



    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu',

                     input_shape=(75, 75, 3)))

    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Dropout(0.2))



    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Dropout(0.2))



    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Dropout(0.3))



    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Dropout(0.3))



    model.add(Flatten())

    model.add(Dense(512, activation='relu'))

    model.add(Dropout(0.2))



    model.add(Dense(256, activation='relu'))

    model.add(Dropout(0.2))



    model.add(Dense(1, activation="sigmoid"))



    return model
def Vgg16():

    input_tensor = Input(shape=(75, 75, 3))

    vgg16 = VGG16(include_top=False, weights='imagenet',

                  input_tensor=input_tensor)



    top_model = Sequential()

    top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))

    top_model.add(Dense(512, activation='relu'))

    top_model.add(Dropout(0.5))

    top_model.add(Dense(256, activation='relu'))

    top_model.add(Dropout(0.5))

    top_model.add(Dense(1, activation='sigmoid'))



    model = Model(input=vgg16.input, output=top_model(vgg16.output))

    for layer in model.layers[:13]:

        layer.trainable = False



    return model
if __name__ == "__main__":

    x, y = make_df("../input/train.json", "train")

    xtr, xval, ytr, yval = train_test_split(x, y, test_size=0.25,

                                            random_state=7)

    model = SmallCNN()

    #model = Vgg16()

    optimizer = Adam(lr=0.001, decay=0.0)

    model.compile(loss='binary_crossentropy', optimizer=optimizer,

                  metrics=['accuracy'])



    earlyStopping = EarlyStopping(monitor='val_loss', patience=20, verbose=0,

                                  mode='min')

    ckpt = ModelCheckpoint('.model.hdf5', save_best_only=True,

                           monitor='val_loss', mode='min')

    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1,

                                       patience=7, verbose=1, epsilon=1e-4,

                                       mode='min')



    gen = ImageDataGenerator(horizontal_flip=True,

                             vertical_flip=True,

                             width_shift_range=0,

                             height_shift_range=0,

                             channel_shift_range=0,

                             zoom_range=0.2,

                             rotation_range=10)

    gen.fit(xtr)

    model.fit_generator(gen.flow(xtr, ytr, batch_size=32),

                        steps_per_epoch=len(xtr), epochs=1,

                        callbacks=[earlyStopping, ckpt, reduce_lr_loss],

                        validation_data=(xval, yval))



    model.load_weights(filepath='.model.hdf5')

    score = model.evaluate(xtr, ytr, verbose=1)

    print('Train score:', score[0], 'Train accuracy:', score[1])



    xtest, df_test = make_df("../input/test.json", "test")

    pred_test = model.predict(xtest)

    pred_test = pred_test.reshape((pred_test.shape[0]))

    submission = pd.DataFrame({'id': df_test["id"], 'is_iceberg': pred_test})

    submission.to_csv('submission.csv', index=False)