import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from scipy import stats



from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.preprocessing import MinMaxScaler, FunctionTransformer

from sklearn.pipeline import Pipeline

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score



import keras

from keras.models import Model, Sequential, load_model

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, GlobalMaxPool2D

from keras.layers import Activation, Add, ReLU, Flatten, Dropout, BatchNormalization

from keras.layers import ZeroPadding2D, Concatenate

from keras.layers.advanced_activations import LeakyReLU

from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import Adam

from keras import backend as K

from keras.utils import to_categorical

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from keras.applications import ResNet50





plt.rcParams['figure.figsize'] = (12, 8)

plt.rcParams['image.interpolation'] = 'nearest'

plt.rcParams['image.cmap'] = 'gray'

K.set_image_data_format('channels_last')

random_state = 42

image_shape = (28, 28, 1)
source_df = pd.read_csv('../input/Kannada-MNIST/train.csv')

test_df = pd.read_csv('../input/Kannada-MNIST/test.csv')

dig_df = pd.read_csv('../input/Kannada-MNIST/Dig-MNIST.csv')
def get_X_Y(df):

    X = df.loc[:, 'pixel0':'pixel783'].copy()

    Y = df.loc[:, 'label'].copy()

    return (X, Y)



source_X, source_Y = get_X_Y(source_df) 

dig_X, dig_Y = get_X_Y(dig_df)



print('source_X shape {}, source_Y shape {}'.format(source_X.shape, source_Y.shape))
Y_classes = sorted(source_Y.unique())

num_classes = len(Y_classes)

plt.figure()



for y_idx, y_class in enumerate(Y_classes):

    idxs = np.random.choice(np.c_[source_Y.loc[source_Y == y_class].index].flatten(), 10, replace=False)

    for i, idx in enumerate(idxs):

        plt_idx = i * num_classes + y_idx + 1

        plt.subplot(num_classes, num_classes, plt_idx)

        plt.imshow(source_X.iloc[idx].values.reshape(28, 28))

        plt.axis('off')

        if i == 0:

            plt.title(str(y_class))
train_X, valid_X, train_Y, valid_Y = train_test_split(

    source_X, source_Y, test_size=0.2, stratify=source_Y, random_state=random_state)

print('train_X shape {}, train_Y shape {}, valid_X shape {}, valid_Y shape {}'.format(

    train_X.shape, train_Y.shape, valid_X.shape, valid_Y.shape))
def build_cnn(input_shape, classes):

    input_layer = Input(shape=input_shape)



    conv_layer1 = Conv2D(32, kernel_size=(3,3), strides=1, padding='same', input_shape=(28, 28, 1))(input_layer)    

    conv_layer1 = Conv2D(16, kernel_size=(3,3), strides=1, padding='same')(conv_layer1)

    batch_norm_layer1 = BatchNormalization(momentum=0.5, gamma_initializer='uniform')(conv_layer1)

    relu_layer1 = LeakyReLU()(batch_norm_layer1)

    

    max_pool_layer1 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(relu_layer1)

    con_drop_layer1 = Dropout(0.5)(max_pool_layer1)

    

    conv_layer2 = Conv2D(64, kernel_size=(3,3), strides=1, padding='same')(con_drop_layer1)    

    conv_layer2 = Conv2D(32, kernel_size=(3,3), strides=1, padding='same')(conv_layer2)

    batch_norm_layer2 = BatchNormalization(momentum=0.5, gamma_initializer='uniform')(conv_layer2)

    relu_layer2 = LeakyReLU()(batch_norm_layer2)

        

    max_pool_layer2 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(relu_layer2)

    con_drop_layer2 = Dropout(0.5)(max_pool_layer2)

    

    conv_layer3 = Conv2D(128, kernel_size=(3,3), strides=1, padding='same')(con_drop_layer2)

    conv_layer3 = Conv2D(64, kernel_size=(3,3), strides=1, padding='same')(conv_layer3)

    batch_norm_layer3 = BatchNormalization(momentum=0.5, gamma_initializer='uniform')(conv_layer3)

    relu_layer3 = LeakyReLU()(batch_norm_layer3)

    

    max_pool_layer3 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(relu_layer3)

    flatten = Flatten()(max_pool_layer3)  

    dropout_layer1 = Dropout(0.5)(flatten)

    

    dense_layer1 = Dense(256)(dropout_layer1)

    relu_layer4 = LeakyReLU()(dense_layer1)

    dropout_layer2 = Dropout(0.5)(relu_layer4)

    

    dense_layer2 = Dense(256)(dropout_layer2)

    relu_layer5 = LeakyReLU()(dense_layer2)

    dropout_layer3 = Dropout(0.5)(relu_layer5)

    

    output_layer = Dense(10, activation='softmax')(dropout_layer3)



    model = Model(inputs=[input_layer], outputs=[output_layer])

    model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])

    

    return model



cnn_model = build_cnn(image_shape, 10)

cnn_model.summary()
def preproc_images(images):

    return (images / 255.0).astype(np.float64).values.reshape((-1, 28, 28, 1))

    

def agg_result_mode(predictions):

    mode_result = stats.mode(predictions.argmax(axis=2), axis=0)

    result = mode_result.mode.squeeze()    

    return result



def agg_result_mean(predictions):

    return predictions.mean(axis=0).argmax(axis=1)



def show_class_reports(y_true, y_predicted):

    print('accuracy {}'.format(accuracy_score(y_true, y_predicted)))

    print(classification_report(y_true, y_predicted))

    print(confusion_matrix(y_true, y_predicted))



def plot_graphs(history, string):

    plt.plot(history.history[string])

    plt.plot(history.history['val_'+string], '')

    plt.xlabel("Epochs")

    plt.ylabel(string)

    plt.legend([string, 'val_'+string])

    plt.show()    



def get_image_data_gen():

    return ImageDataGenerator(

        rotation_range=15,

        zoom_range=0.15,

        shear_range=0.15,

        width_shift_range=0.15,

        height_shift_range=0.15

    )



def submit(model, kfold=False):

    test_ID = test_df.loc[:, 'id']

    test_X = test_df.loc[:, 'pixel0':'pixel783']

    test_X = preproc_images(test_X)

    

    if kfold:

        test_Ys = np.array([m.predict(test_X) for m in model])

        test_Y = agg_result_mean(test_Ys)    

    else:

        test_Y = model.predict(test_X)

        test_Y = np.argmax(test_Y, axis=1)



    submition_df = pd.DataFrame.from_dict({ 'id': test_ID, 'label': test_Y })

    submition_df.to_csv('./submission.csv', index=False)





def run_model(model, X, y, batch_size, number=0, **kwards):

    best_model_path = 'best_model_{}.h5'.format(number)

    

    image_data_gen = get_image_data_gen()

    image_data_gen.fit(X)

    

    callbacks = [

        EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience=15, verbose=1, mode='auto'),

        ModelCheckpoint(best_model_path, monitor='val_accuracy', verbose=1, mode='auto', save_best_only=True),

        ReduceLROnPlateau(monitor='val_accuracy', factor=0.7, patience=2, verbose=1, mode='auto')

    ]

    

    images_flow = image_data_gen.flow(X, y, batch_size=batch_size)  

    steps_per_epoch = X.shape[0] // batch_size

    hist = model.fit(images_flow, steps_per_epoch=steps_per_epoch, callbacks=callbacks, **kwards)

    model.load_weights(best_model_path)



    return hist
model_count = 6

epochs = 25

batch_size = 128



models = [build_cnn(image_shape, num_classes) for _ in range(model_count)]

kfold_splitter = StratifiedKFold(n_splits=model_count, shuffle=True, random_state=random_state)

histories = []



for index, (train_index, valid_index) in enumerate(kfold_splitter.split(source_X, source_Y)):

    print('Start model {} training'.format(index + 1))

    

    k_fold_train_X, k_fold_train_Y = source_X.iloc[train_index], source_Y.iloc[train_index]

    k_fold_valid_X, k_fold_valid_Y = source_X.iloc[valid_index], source_Y.iloc[valid_index]

    

    k_fold_train_X_prep, k_fold_valid_X_prep = preproc_images(k_fold_train_X), preproc_images(k_fold_valid_X)

    

    run_model(

        models[index], k_fold_train_X_prep, to_categorical(k_fold_train_Y), batch_size, number=index,

        epochs=epochs, validation_data=(k_fold_valid_X_prep, to_categorical(k_fold_valid_Y))

    )

for hist in histories:

    plot_graphs(hist, 'accuracy')
submit(models, kfold=True)