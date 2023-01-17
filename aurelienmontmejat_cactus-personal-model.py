import pandas as pd

import matplotlib.pyplot as plt

import os, random

import numpy as np

from PIL import Image

from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras import Sequential

from tensorflow.keras.layers import *
train_df = pd.read_csv("../input/train.csv", dtype=str)

test_df = pd.read_csv("../input/sample_submission.csv", dtype=str)

test_files_df = test_df.drop('has_cactus', axis=1)

#print(train_df)
datagen = ImageDataGenerator(rescale=1./255)



train_generator = datagen.flow_from_dataframe(

        dataframe=train_df[:15001],

        directory='../input/train/train/',

        x_col='id',

        y_col='has_cactus',

        shuffle=True,

        class_mode='binary',

        batch_size=150,

        target_size=(150, 150))



validation_generator = datagen.flow_from_dataframe(

        dataframe=train_df[15000:],

        directory='../input/train/train/',

        x_col='id',

        y_col='has_cactus',

        class_mode='binary',

        batch_size=50,

        target_size=(150, 150))
model = Sequential([

    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),

    MaxPool2D(2, 2),

    Dropout(0.25),

    

    Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),

    MaxPooling2D(2, 2),

    Dropout(0.25),

    

    Conv2D(128, (3,3), activation='relu', input_shape=(150, 150, 3)),

    MaxPooling2D(2, 2),

    Dropout(0.25),

    

    Conv2D(128, (3,3), activation='relu', input_shape=(150, 150, 3)),

    MaxPooling2D(2, 2),

    Dropout(0.25),

    

    Flatten(),

    Dense(512, activation='relu'),

    Dropout(0.5),

    Dense(1, activation='sigmoid')

])



model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

history = model.fit_generator(generator=train_generator, steps_per_epoch=100, epochs=10, validation_data=validation_generator, validation_steps=50)
epochs = 10



accuracy = history.history['acc']  # getting accuracy of each epochs

epochs_ = range(0, epochs)    

plt.plot(epochs_, accuracy, label='training accuracy')

plt.xlabel('no of epochs')

plt.ylabel('accuracy')



acc_val=history.history['val_acc']  # getting validation accuracy of each epochs

plt.scatter(epochs_,acc_val,label="validation accuracy")

plt.title("no of epochs vs accuracy")

plt.legend()
test_generator = datagen.flow_from_dataframe(

        dataframe=test_files_df,

        directory='../input/test/test/',

        x_col='id',

        class_mode=None,

        shuffle=False,

        target_size=(150, 150))



predict = model.predict_generator(test_generator)



for i in range(0, 5):

    rand = random.randint(0, len(predict))

    print("Prediction for image " + str(rand) + ": " + str(predict[rand]))

    img_array = np.array(Image.open(os.path.join("../input/test/test", test_df.iloc[rand, 0])))

    plt.imshow(img_array)

    plt.show()
for i in range(0, len(predict)):

    test_df.at[i, 'has_cactus'] = 0 if predict[i] < 0.5 else 1

    

test_df.to_csv('submission.csv', index=False)