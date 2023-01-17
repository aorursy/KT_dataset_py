import numpy as np
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Dense, Activation, Dropout, Flatten
from keras import optimizers
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
IMG_SIZE = 100
LEARNING_RATE = 0.5
EPOCHS = 5
BATCH_SIZE = 32

TRAIN_SIZE = 8005
TEST_SIZE = 2023

TRAIN_DIR = '../input/cat-and-dog/training_set/training_set'
TEST_DIR = '../input/cat-and-dog/test_set/test_set'
augmented_gen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

print("Training Set")
train_gen = augmented_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    classes=['dogs', 'cats'],
    batch_size=BATCH_SIZE)
print("")

print("Test Set")
test_gen = augmented_gen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    classes=['dogs', 'cats'],
    batch_size=BATCH_SIZE)
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

model.summary()
training = model.fit_generator(
    generator=train_gen,
    steps_per_epoch= TRAIN_SIZE // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=test_gen,
    validation_steps= TEST_SIZE // BATCH_SIZE)
TEST_IMG_PATH = "../input/cat-and-dog/test_set/test_set/dogs/dog.4006.jpg"

def predict(img_path):
    img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    x = x/255
    result = model.predict(x).tolist()

    if result[0][0] >= 0.5:
        prediction = 'CAT 🐱'
        confidence = ("{:.2%}".format(result[0][0]))
    else:
        prediction = 'DOG 🐶'
        confidence = ("{:.2%}".format(result[0][1]))
        
    print("I am {0} confident that this is a {1}".format(confidence, prediction))
    
predict(TEST_IMG_PATH)