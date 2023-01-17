import numpy as np

import pandas as pd

from keras.layers import *

from keras.models import *

from keras.optimizers import *

from keras.preprocessing.image import ImageDataGenerator

import os
def load_datasets(validation_split=0.2):

    def load_data(filepath, test=False):

        df = pd.read_csv(filepath)

        data = df.values

        if not test:

            x_data = data[:, 1:]

            y_data = data[:, 0]

        else: 

            x_data = data

            y_data = np.empty(0)

        x_data = x_data.astype(np.float32) / 255.0

        x_data = np.reshape(x_data, (x_data.shape[0], 28, 28, 1))



        return x_data, y_data



    base_folder = '../input/'



    x_train = []

    y_train = []

    x_test = []



    x_train, y_train = load_data(base_folder + 'train.csv')

    x_test, _ = load_data(base_folder + 'test.csv', test=True)



    indexes = np.random.permutation(range(x_train.shape[0]))

    x_train = x_train[indexes]

    y_train = y_train[indexes]



    split_index = int(x_train.shape[0] * validation_split)



    x_val = x_train[:split_index]

    y_val = y_train[:split_index]



    x_train = x_train[split_index:]

    y_train = y_train[split_index:]

    

    return x_train, y_train, x_val, y_val, x_test





def build_model():

    kernel_size = 3

    filters = 16

    image_size = 28

    classes = 10



    model_inputs = Input((28, 28,1))

    processed = model_inputs



    while image_size != 3:

        processed = Conv2D(filters, kernel_size, padding='same')(processed)

        processed = LeakyReLU()(processed) 

        processed = Conv2D(filters, kernel_size, padding='same')(processed)

        processed = LeakyReLU()(processed)  



        processed = MaxPooling2D()(processed)



        image_size = int(image_size / 2)

        filters *= 2    



    processed = GlobalAveragePooling2D()(processed)

    processed = Dropout(0.5)(processed)

    processed = Dense(10, activation='softmax')(processed)



    model = Model(model_inputs, processed)

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print(model.summary())

    return model

    

    

def save_predictions(model, x_test):

    predictions = model.predict(x_test)

    predictions = np.argmax(predictions, axis=1)



    with open('output.csv', 'w') as f:

        print('ImageId,Label', file=f)



        for i, prediction in enumerate(predictions):

            print(str(i+1) + ',' + str(prediction), file=f)
x_train, y_train, x_val, y_val, x_test = load_datasets()

print(x_train.shape, y_train.shape, x_val.shape, y_val.shape, x_test.shape)



datagen = ImageDataGenerator(

    rotation_range=10,

    width_shift_range=0.1,

    height_shift_range=0.1,

    zoom_range = 0.1)

datagen.fit(x_train)
model = build_model()



batch_size = 32

model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),

                    validation_data=(x_val, y_val),

                    steps_per_epoch=x_train.shape[0] / batch_size, epochs=15)
save_predictions(model, x_test)