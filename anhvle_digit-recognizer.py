



import numpy as np

import pandas as pd

from PIL import Image

import os

import tensorflow as tf

import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMAGE_SIZE = 28

TRAIN_PATH = '../input/digit-recognizer/train.csv'

TEST_PATH = '../input/digit-recognizer/test.csv'
def show_img(img, title=''):

    # given a numpy array, plot it

    vis = plt.imshow(img)

    plt.title(title)

    plt.show()
def parse_line(line):

    # given an input line, transform the data into image and label

    label = line['label'].iloc[0]

    img_array = line.loc[:, line.columns != 'label']

    img_array = np.array(img_array).astype('float32')

    img_array.resize((28, 28))

    img = Image.fromarray(test_image)

    return label, img
train_lines = pd.read_csv(TRAIN_PATH)

test_lines = pd.read_csv(TEST_PATH)
test_label, test_img = parse_line(train_lines.loc[[0]])

show_img(test_img, title='label: ' + str(test_label))
x_train = (train_lines.iloc[:,1:].values).astype('float32')

y_train = (train_lines.iloc[:,0].values).astype('int32')

x_test = (test_lines.iloc[:].values).astype('float32')
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
y_train = tf.keras.utils.to_categorical(y_train)

y_train.shape
train_datagen = ImageDataGenerator(

    validation_split=0.2,

    rotation_range=20,

)



train_generator = train_datagen.flow(x_train, y_train, seed=1, subset='training')

validation_generator = train_datagen.flow(x_train, y_train, seed=1, subset='validation')
model = tf.keras.models.Sequential([

    # convolutions and down sampling

    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1), padding='same'),

    tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(64),

    tf.keras.layers.Dense(32),

    tf.keras.layers.Dense(10, activation='softmax')

])



model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

model.summary()
callbacks = []
print(x_test.shape)

print(x_train.shape)

print(y_train.shape)

print(model.output_shape)
model_history = model.fit(

    train_generator,

    epochs=20,

    validation_data=validation_generator,

    validation_steps=int((y_train.shape[0] * 0.2) / 32),

    callbacks=callbacks,

    verbose=1

)
plt.plot(model_history.history['acc'])

plt.plot(model_history.history['val_acc'])

plt.title('Model Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Training set', 'Validation set'], loc='upper left')

plt.show()



# Loss 



plt.plot(model_history.history['val_loss'])

plt.plot(model_history.history['loss'])

plt.title('Model Loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Training set', 'Test set'], loc='upper left')

plt.show()
predictions = model.predict_classes(x_test)
predictions[0]
submissions=pd.DataFrame({

    'ImageId': list(range(1,len(predictions)+1)),

    'Label': predictions

})
submissions.to_csv('predictions.csv', index=False, header=True)