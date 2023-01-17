import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
random_seed = 1213

np.random.seed(random_seed)



from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from tensorflow.keras.layers import Input, concatenate, BatchNormalization, Add, Activation

from tensorflow.keras.optimizers import RMSprop, Adam

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from tensorflow.keras.utils import plot_model



from sklearn.model_selection import train_test_split



import matplotlib.pyplot as plt
#Load train and test data

train = pd.read_csv('../input/ciproject/persianMNIST_train.csv')

test = pd.read_csv('../input/ciproject/persianMNIST_test.csv')



#Extract and separate prediction (Y) and Inputs (X)

Y = train['0']

X = train.drop(labels="0", axis=1)



#Reshape: Original image is a 28x28 pixel image

X = X.values.reshape(-1, 28, 28, 1)

test = test.values.reshape(-1, 28, 28, 1)



#Separate training data and validation data

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.10, random_state=random_seed)

print(X.shape, test.shape)
learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_acc',

                                           patience = 3,

                                           verbose = 1,

                                           factor = 0.5,

                                           min_lr = 0.0001)



es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)



def new_model(hidden=512, learning_rate=0.00128):

    INPUT   = Input((28, 28, 1))

    #First Convolution

    inputs  = Conv2D(64, (7, 7), activation='relu', padding='same')(INPUT)

    inputs  = MaxPool2D(pool_size=(5,5), strides=(2,2))(inputs)

    inputs  = BatchNormalization()(inputs)

    inputs  = Activation('relu')(inputs)

    inputs  = Dropout(0.5)(inputs)

    #Branch off to Three Towers

    #First Tower

    tower_1 = Conv2D(64, (1, 1), activation='relu', padding='same')(inputs)

    tower_1 = Conv2D(64, (2, 2), activation='relu', padding='same')(tower_1)

    tower_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(tower_1)

    tower_1 = MaxPool2D(pool_size=(3,3), strides=(2,2))(tower_1)

    tower_1 = BatchNormalization()(tower_1)

    #Second Tower

    tower_2 = Conv2D(64, (2, 2), activation='relu', padding='same')(inputs)

    tower_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(tower_2)

    tower_2 = Conv2D(64, (5, 5), activation='relu', padding='same')(tower_2)

    tower_2 = MaxPool2D(pool_size=(3,3), strides=(2,2))(tower_2)

    tower_2 = BatchNormalization()(tower_2)

    #Third Tower

    tower_3 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)

    tower_3 = Conv2D(64, (3, 3), activation='relu', padding='same')(tower_3)

    tower_3 = Conv2D(64, (5, 5), activation='relu', padding='same')(tower_3)

    tower_3 = MaxPool2D(pool_size=(3,3), strides=(2,2))(tower_3)

    tower_3 = BatchNormalization()(tower_3)

    #Combine Three Towers

    x       = Add()([tower_1, tower_2, tower_3])

    x       = Activation('relu')(x)

    #Last Convolution

    x       = Conv2D(128, (5, 5), activation='relu', padding='same')(x)

    x       = MaxPool2D(pool_size=(5,5), strides=(3,3))(x)

    x       = BatchNormalization()(x)

    x       = Activation('relu')(x)

    #Flatten Data

    x       = Flatten()(x)

    #Dense Hidden Network

    x       = Dense(hidden, activation='relu')(x)

    x       = Dropout(0.5)(x)

    #Model Output

    preds   = Dense(10, activation='softmax', name='preds')(x)

    #Build Model

    model   = Model(inputs=INPUT, outputs=preds)

    #Define Optimizer

    optimizer = Adam(lr=learning_rate)

    #Compile model

    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['acc'])

    

    return model



model = new_model()
model.summary()

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
#Data Augmentation to prevent overfitting

datagen = ImageDataGenerator(featurewise_center=False,

                            samplewise_center=False,

                            featurewise_std_normalization=False,

                            samplewise_std_normalization=False,

                            zca_whitening=False,

                            rotation_range=5,

                            zoom_range=0.05,

                            shear_range=0.02,

                            width_shift_range=0.05,

                            height_shift_range=0.05,

                            horizontal_flip=False,

                            vertical_flip=False)
epochs = 200

batch_size = 128



print("Learning Properties: Epoch:%i \t Batch Size:%i" %(epochs, batch_size))

predict_accumulator = np.zeros(model.predict(test).shape)



accumulated_history = []

for i in range(1, 6):

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.20, random_state=random_seed*i)

    model = new_model(100, 0.008/i)

    #Fit the model

    datagen.fit(X_train)

    history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),

                                 epochs=epochs, validation_data=(X_val, Y_val), verbose=3,

                                 steps_per_epoch=X_train.shape[0]//batch_size,

                                 callbacks=[learning_rate_reduction, es],

                                 workers=4)

    loss, acc = model.evaluate(X, Y)

    if acc > 0.75:

        predict_accumulator += model.predict(test)*acc

        accumulated_history.append(history)

        print("Current Predictions on fold number %i", i)

        print(*np.argmax(predict_accumulator, axis=1), sep='\t')
def graph(full_history):

    '''Show and save the historical graph of the training model.'''

    print('Accuracy:')

    for history in full_history:

        plt.plot(history.history['acc'])

        plt.plot(history.history['val_acc'])

    plt.title('Model Accuracy')

    plt.ylabel('Accuracy')

    plt.xlabel('Epoch')

    plt.legend(['train', 'test'], loc='lower right')

    plt.savefig('history_acc.png')

    plt.show()



    print('Loss:')

    for history in full_history:

        plt.plot(history.history['loss'])

        plt.plot(history.history['val_loss'])

    plt.title('Model Loss')

    plt.ylabel('Loss')

    plt.xlabel('Epoch')

    plt.legend(['train', 'test'], loc='upper right')

    plt.savefig('history_loss.png')

    plt.show()



graph(accumulated_history)
print("Completed Training.")

results = np.argmax(predict_accumulator, axis=1)

results = pd.Series(results, name="result")

print("Saving prediction to output...")

submission = pd.concat([pd.Series(range(0, test.shape[0]), name="ids"), results], axis=1)

submission.to_csv('prediction.csv', index=False)