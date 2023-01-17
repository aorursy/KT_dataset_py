import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from time import time

print('loading data file...')
t0 = time()
data = pd.read_csv("../input/data.csv")
print("loaded in %fs" % (time() - t0))
data.head()

X = data.iloc[:, :-1]
y = data.iloc[:, -1]
print('Checking lengths:\ndata :', len(data), '\nX :', len(X), '\ny :', len(y))

classes_list = np.unique(y)
class_to_int = {cur_class: i for i, cur_class in enumerate(classes_list)}
nb_classes = len(classes_list)

X_flat = X.values.reshape(len(X), -1)
y_num = [class_to_int[i] for i in y]

print('Checking pictures range of values:\nMin :', np.min(X_flat), '\nMax :', np.max(X_flat))
print('Checking target classes:\n', classes_list)
print('Checking numerical targets:\n', np.unique(y_num))

X_flat = (255 - X_flat) / 255
X_flat = X_flat.astype(np.float32)
y_num = np.array(y_num).reshape(-1, 1).astype(np.float32)

print('Checking y_num\'s shape', y_num.shape)

X_train, X_test, y_train, y_test = train_test_split(X_flat, y_num, test_size=0.2, random_state=0)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

image_border = 32
image_shape = (image_border, image_border)


def visualise_samples():
    X_flat_copy = X_flat.copy()
    np.random.shuffle(X_flat_copy)
    for i in range(1, 9):
        plt.subplot(240 + i)
        plt.axis('off')
        image = X_flat_copy[i-1].reshape(image_shape)
        plt.imshow(image, cmap=plt.cm.gray)
    plt.show()
    del X_flat_copy


visualise_samples()
mean_images = {}
for i, current_class in enumerate(classes_list):
    print('Computing mean picture for class', current_class)
    composed_img = np.zeros(image_shape)
    cnt = 0
    mask = y_train.reshape(-1) == i
    for picture in X_train[mask,:]:
        picture = np.array(picture).reshape(image_shape)
        composed_img += picture.astype(np.float32)
        cnt += 1
    composed_img = composed_img - np.min(composed_img)
    composed_img = composed_img / np.max(composed_img)
    mean_images[i] = composed_img

def plot_mean_images(title, images, image_shape, n_col=10, n_row=10):
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    plt.suptitle(title, size=16)
    for i in images:
        comp = images[i]
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(comp.reshape(image_shape), cmap=plt.cm.gray)
        plt.title(i)
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.25, 0.50)

plot_mean_images("Mean pictures", mean_images, (image_border, image_border), 8, 8)
def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


models = np.zeros((len(classes_list), image_border, image_border, 1))

for i, current_class in enumerate(classes_list):
    print('Choosing best picture for class', current_class)
    mean_image = mean_images[i].reshape((image_border, image_border, 1))
    min_ = 100000000000
    best_img = None
    mask = y_train.reshape(-1) == i
    for picture in X_train[mask, :]:
        picture = np.array(picture).reshape((1, image_border, image_border, 1))
        sum_ = mse(mean_image, picture)
        if sum_ < min_:
            min_ = sum_
            best_img = picture
    models[i, :, :, :] = best_img


def plot_models(title, images, image_shape, n_col=10, n_row=10):
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    plt.suptitle(title, size=16)
    for i, image in enumerate(images):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(image.reshape(image_shape), cmap=plt.cm.gray)
        plt.title(i)
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.25, 0.50)


plot_models("Models", models, (image_border, image_border), 8, 8)
import cv2

X_train_shape = (len(X_train), image_border, image_border, 1)
arr = np.zeros(X_train_shape)
for i, y in enumerate(y_train):
    arr[i, :, :, :] = models[int(y), :, :, :]
models_train = arr

X_valid_shape = (len(X_valid), image_border, image_border, 1)
arr = np.zeros(X_valid_shape)
for i, y in enumerate(y_valid):
    arr[i, :, :, :] = models[int(y), :, :, :]
models_valid = arr

X_test_shape = (len(X_test), image_border, image_border, 1)
arr = np.zeros(X_test_shape)
for i, y in enumerate(y_test):
    arr[i, :, :, :] = models[int(y), :, :, :]
models_test = arr

del arr

models_train = models_train.astype('float32')
models_valid = models_valid.astype('float32')
models_test = models_test.astype('float32')

X_train = X_train.reshape(X_train_shape)
X_valid = X_valid.reshape(X_valid_shape)
X_test = X_test.reshape(X_test_shape)

print('X_train min', np.min(X_train[0]), ' max', np.max(X_train[0]))
print('X_valid min', np.min(X_valid[0]), ' max', np.max(X_valid[0]))
print('X_test min', np.min(X_test[0]), ' max', np.max(X_test[0]))
print('models_train min', np.min(models_train[0]), ' max', np.max(models_train[0]))
print('models_valid min', np.min(models_valid[0]), ' max', np.max(models_valid[0]))
print('models_test min', np.min(models_test[0]), ' max', np.max(models_test[0]))


def visualize_training_couples(title):
    n_col = int(np.ceil(np.sqrt(image_border)))
    n_row = int(np.ceil(np.sqrt(image_border)))
    plt.figure(figsize=(6. * n_col, 6.26 * n_row))
    plt.suptitle(title, size=16)
    for i in range(0, 6 * 4, 6):  # enumerate(X_train[:(n_col * n_row) / 3]):
        plt.subplot(n_row, n_col, i + 1)
        pic = X_train[i]
        plt.imshow(pic.reshape(image_shape), cmap=plt.cm.gray,
                   interpolation='nearest')
        plt.subplot(n_row, n_col, i + 2)
        pic = models_train[i]
        plt.imshow(pic.reshape(image_shape), cmap=plt.cm.gray,
                   interpolation='nearest')
        plt.subplot(n_row, n_col, i + 3)
        pic = X_valid[i]
        plt.imshow(pic.reshape(image_shape), cmap=plt.cm.gray,
                   interpolation='nearest')
        plt.subplot(n_row, n_col, i + 4)
        pic = models_valid[i]
        plt.imshow(pic.reshape(image_shape), cmap=plt.cm.gray,
                   interpolation='nearest')
        plt.subplot(n_row, n_col, i + 5)
        pic = X_test[i]
        plt.imshow(pic.reshape(image_shape), cmap=plt.cm.gray,
                   interpolation='nearest')
        plt.subplot(n_row, n_col, i + 6)
        pic = models_test[i]
        plt.imshow(pic.reshape(image_shape), cmap=plt.cm.gray,
                   interpolation='nearest')
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)


visualize_training_couples('Training couples')
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, BatchNormalization
from keras.models import Model

def train_model():
    input_img = Input(shape=(image_border, image_border, 1))
    # layer shape 32 x 32
    x = Conv2D(64, (9, 9), activation='relu', padding='same')(input_img)
    x = BatchNormalization()(x)
    x = Conv2D(64, (9, 9), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    # layer shape 16 x 16
    x = Conv2D(64, (9, 9), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    # layer shape 8 x 8
    x = Conv2D(64, (7, 7), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    encoded = BatchNormalization(name='encoder')(x)
    # layer shape 4 x 4
    # at this point the representation is (4, 4, 64) i.e. 1024-dimensional
    x = UpSampling2D((2, 2))(encoded)
    # layer shape 8 x 8
    x = Conv2D(64, (7, 7), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    # layer shape 16 x 16
    x = Conv2D(64, (9, 9), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    # layer shape 32 x 32
    x = Conv2D(64, (9, 9), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    decoded = Conv2D(1, (9, 9), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.summary()
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy',
                        metrics=['accuracy'])
    autoencoder.fit(X_train, models_train,
                    epochs=10,
                    batch_size=32,
                    shuffle=True,
                    verbose=1,
                    validation_data=(X_valid, models_valid))

    autoencoder.save('Devanagari_autoencoder.h5')
    score = autoencoder.evaluate(X_test, models_test)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    return autoencoder


autoencoder = train_model()
from keras.models import load_model

def encode_and_save(autoencoder=None):
    print('Loading model :')
    t0 = time()
    # Load previously trained autoencoder
    if autoencoder is None:
        autoencoder = load_model('Devanagari_autoencoder.h5')
    print('Model loaded in: ', time() - t0)
    
    encoder = Model(inputs=autoencoder.input,
                    outputs=autoencoder.get_layer('encoder').output)
    
    print('Encoding pictures...')
    batch_size = 32
    encoded_train = encoder.predict(X_train, batch_size=batch_size, verbose=1)
    encoded_valid = encoder.predict(X_valid, batch_size=batch_size, verbose=1)
    encoded_test = encoder.predict(X_test, batch_size=batch_size, verbose=1)
    
    # encoded_train = pd.DataFrame(encoded_train.reshape(X_train.shape[0], -1))
    # encoded_train.to_csv('Encoded_X_train.csv', header=False, index=False)

    # encoded_valid = pd.DataFrame(encoded_valid.reshape(X_valid.shape[0], -1))
    # encoded_valid.to_csv('Encoded_X_valid.csv', header=False, index=False)

    # encoded_test = pd.DataFrame(encoded_test.reshape(X_test.shape[0], -1))
    # encoded_test.to_csv('Encoded_X_test.csv', header=False, index=False)

    return encoded_train, encoded_valid, encoded_test

encoded_train, encoded_valid, encoded_test = encode_and_save(autoencoder)
from keras.utils import to_categorical
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras import backend as K
import keras


def plot_history(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    plt.figure()
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def plot_gallery(title, images, image_shape):
    n_col = int(np.ceil(np.sqrt(images.shape[0])))
    n_row = int(np.ceil(np.sqrt(images.shape[0])))
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    plt.suptitle(title, size=16)
    for i, comp in enumerate(images[:(n_col * n_row)]):
        plt.subplot(n_row, n_col, i + 1)
        vmax = max(comp.max(), -comp.min())
        plt.imshow(comp.reshape(image_shape), cmap=plt.cm.gray,
                   interpolation='nearest',
                   vmin=-vmax, vmax=vmax)
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)


n_row, n_col = 30, 25


def plot_gallery_2(title, images, image_shape, predicted_class=None,
                   predictions=None, targets=None, n_col=n_col, n_row=n_row):
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    plt.suptitle(title, size=16)
    for i, comp in enumerate(images[:(n_col * n_row)]):
        plt.subplot(n_row, n_col, i + 1)
        vmax = max(comp.max(), -comp.min())
        plt.imshow(comp.reshape(image_shape), cmap=plt.cm.gray,
                   interpolation='nearest',
                   vmin=-vmax, vmax=vmax)
        idx_sort = np.argsort(predictions[i])[::-1]
        if predicted_class is not None and predictions is not None and targets is not None:
            first_guess = idx_sort[0]
            second_guess = idx_sort[1]
            third_guess = idx_sort[2]
            fourth_guess = idx_sort[3]
            true = targets[i]
            display = str(true)
            display += '/' + str(first_guess)
            display += '-' + str(second_guess)
            display += '-' + str(third_guess)
            display += '-' + str(fourth_guess)
            plt.title(display)
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.25, 0.50)


def train_dense(encoded_train, encoded_valid, encoded_test ,X_train, X_valid, X_test):
    y_train_categorical = to_categorical(y_train).astype(np.float32)
    y_valid_categorical = to_categorical(y_valid).astype(np.float32)
    y_test_categorical = to_categorical(y_test).astype(np.float32)

    img_rows, img_cols = image_shape
    nb_channels = 1
    if K.image_data_format() == 'channels_first':
        X_train = X_train.reshape(X_train.shape[0], nb_channels, img_rows, img_cols)
        X_valid = X_valid.reshape(X_valid.shape[0], nb_channels, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], nb_channels, img_rows, img_cols)
        input_shape = (64, 4, 4)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, nb_channels)
        X_valid = X_valid.reshape(X_valid.shape[0], img_rows, img_cols, nb_channels)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, nb_channels)
        input_shape = (4, 4, 64)

    dense_input = Input(shape=input_shape)
    x = Dense(nb_classes, activation='relu', input_shape=input_shape)(dense_input)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    output = Dense(nb_classes, activation='softmax')(x)
    dense_model = Model(dense_input, output)

    dense_model.summary()

    dense_model.compile(loss=keras.losses.categorical_crossentropy,
                        optimizer=keras.optimizers.Adadelta(),
                        metrics=['accuracy'])

    print('Encoding pictures...')
    if encoded_train is None:
        df_encoded_train = pd.read_csv('Encoded_X_train.csv', header=None)
        encoded_train = np.array(df_encoded_train)
        encoded_train = encoded_train.reshape((len(df_encoded_train), ) + input_shape)
        encoded_train = encoded_train.astype(np.float32)
        del df_encoded_train
    if encoded_valid is None:
        df_encoded_valid = pd.read_csv('Encoded_X_valid.csv', header=None)
        encoded_valid = np.array(df_encoded_valid)
        encoded_valid = encoded_valid.reshape((len(df_encoded_valid), ) + input_shape)
        encoded_valid = encoded_valid.astype(np.float32)
        del df_encoded_valid
    if encoded_test is None:
        df_encoded_test = pd.read_csv('Encoded_X_test.csv', header=None)
        encoded_test = np.array(df_encoded_test)
        encoded_test = encoded_test.reshape((len(df_encoded_test), ) + input_shape)
        encoded_test = encoded_test.astype(np.float32)
        del df_encoded_test


    batch_size = 32
    epochs = 2
    history = dense_model.fit(encoded_train, y_train_categorical,
                              batch_size=batch_size,
                              epochs=epochs,
                              verbose=1,
                              validation_data=(encoded_valid, y_valid_categorical))

    plot_history(history)

    dense_model.save('Devanagari_dense.h5')

    predictions = dense_model.predict(encoded_test, batch_size, verbose=1)

    predicted_class = np.argmax(predictions, axis=1)

    mask = predicted_class != np.squeeze(y_test)

    wrong_guesses_images = X_test[mask]

    wrong_guesses_target = encoded_test[mask]

    wrong_guesses_predictions = predictions[mask]

    wrong_guesses_class = predicted_class[mask]

    good_labels = y_test[mask]

    plot_gallery_2("Wrong guesses", wrong_guesses_images, image_shape,
                   wrong_guesses_class, wrong_guesses_predictions, good_labels,
                   15, 15)

    plot_gallery_2("Wrong guesses target", wrong_guesses_target, image_shape,
                   wrong_guesses_class, wrong_guesses_predictions, good_labels,
                   15, 15)

    score = dense_model.evaluate(encoded_test, y_test_categorical, verbose=1)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    return dense_model

dense = train_dense(encoded_train, encoded_valid, encoded_test, X_train, X_valid, X_test)
def compile_model(autoencoder=None, dense=None):
    if autoencoder is None:
        autoencoder = load_model('Devanagari_autoencoder.h5')

    encoder = Model(inputs=autoencoder.input,
                    outputs=autoencoder.get_layer('encoder').output)
    
    if dense is None:
        dense = load_model('Devanagari_dense.h5')
    
    model = Sequential()
    model.add(encoder)
    model.add(dense)
    model.save('Devanagari_CAE.h5')

compile_model(autoencoder, dense)