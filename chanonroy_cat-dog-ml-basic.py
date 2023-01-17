import tensorflow as tf
import numpy as np
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Dense, Activation, Dropout, Flatten
from keras import optimizers
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

using_gpu = tf.test.is_gpu_available()

if using_gpu:
    print("Using GPU - performance boost activated")
else:
    print("Not using GPU - performance may be slower")
IMG_SIZE = 100

BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.5

TRAIN_SIZE = 8005
TEST_SIZE = 2023

TRAIN_DIR = '../input/cat-and-dog/training_set/training_set'
TEST_DIR = '../input/cat-and-dog/test_set/test_set'

print("Hyperparameters set")
datagen = ImageDataGenerator(rescale=1./255)

print("Training Set")
train_gen = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    classes=['dogs', 'cats'],
    batch_size=BATCH_SIZE)
print("")

print("Test Set")
test_gen = datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    classes=['dogs', 'cats'],
    batch_size=BATCH_SIZE)
from keras.utils.vis_utils import plot_model

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(IMG_SIZE, IMG_SIZE, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), input_shape=(IMG_SIZE, IMG_SIZE, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), input_shape=(IMG_SIZE, IMG_SIZE, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

print("Visualizing model...")
model.summary()
training = model.fit_generator(
    generator=train_gen,
    steps_per_epoch= TRAIN_SIZE // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=test_gen,
    validation_steps= TEST_SIZE // BATCH_SIZE)
import matplotlib.pyplot as plt  

fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(16,6))

# loss history
ax0.plot(training.history['loss'])
ax0.plot(training.history['val_loss'])
ax0.set_title('Model Loss (lower is better)')
ax0.set_ylabel('Loss')
ax0.set_xlabel('Epoch')
ax0.legend(['Train', 'Test'], loc='upper right')

#accuracy history
ax1.plot(training.history['accuracy'])
ax1.plot(training.history['val_accuracy'])
ax1.set_title('Model Accuracy (higher is better)')
ax1.set_ylabel('Accuracy')
ax1.set_xlabel('Epoch')
ax1.legend(['Train', 'Test'], loc='upper left')

plt.show()
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def predict(img_path):
    # show image
    show_image = mpimg.imread(TEST_IMG_PATH)
    imgplot = plt.imshow(show_image)

    # resize image and turn into array
    img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    x = x/255
    result = model.predict(x).tolist()

    # make a prediction with confidence
    if result[0][0] >= 0.5:
        prediction = 'DOG 🐶'
        confidence = ("{:.2%}".format(result[0][0]))
    else:
        prediction = 'CAT 🐱'
        confidence = ("{:.2%}".format(result[0][1]))
        
    print("I am {0} confident that this is a {1}".format(confidence, prediction))
    
print("Prediction function ready")
TEST_IMG_PATH = "../input/cat-and-dog/test_set/test_set/cats/cat.4015.jpg"

predict(TEST_IMG_PATH)