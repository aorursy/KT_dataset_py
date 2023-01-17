import os
import numpy as np
import matplotlib.pyplot as plt

from random import shuffle
from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img, load_img
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import to_categorical

TRAIN_DIR = '../input/train'
TEST_DIR = '../input/test'

image_width = 160
image_height = 160
pool_size = 2
learning_rate = 1e-4
batch_size = 32
epochs = 20

classes = {'dog': 1, 'cat': 0}

import random 
random.seed(1)
def read_image(dir_path, file_name):
    image_path = os.path.join(dir_path, file_name)
    raw_image = load_img(image_path, target_size=(image_width, image_height))
    if raw_image is not None:
        img_array = img_to_array(raw_image)
        return img_array

def display_image(image, prediction=None):
    label_name = None
    accuracy = 0.0
    if prediction >= 0.5:
        label_name = 'Dog'
        accuracy = prediction
    else:
        label_name = 'Cat'
        accuracy = 1-prediction
    
    plt.title('label: {}, accuracy: {:.2%}'.format(label_name, accuracy))
    plt.imshow(array_to_img(image))
    plt.show()

def load_train_data(is_train=True):
    file_list = os.listdir(TRAIN_DIR)
    shuffle(file_list)
    
    file_list = file_list[:20000]
    bucket_size = int(len(file_list)*0.9)
    if is_train:
        file_list = file_list[:bucket_size]
    else:
        file_list = file_list[bucket_size:]

    features = []
    labels = []
    for file_name in tqdm(file_list):
        # 1 = dog, 0 = cat
        label = classes[file_name.split('.')[0]]
        image = read_image(TRAIN_DIR, file_name)
        if image is not None and label is not None:
            features.append(image)
            labels.append(label)
    return np.array(features), labels

def load_test_data():
    test_data = []
    names = []
    for file_name in tqdm(os.listdir(TEST_DIR)):
        image = read_image(TEST_DIR, file_name)
        if image is not None:
            test_data.append(image)
            names.append(file_name.split('.')[0])
    return np.array(test_data), names

X_train, y_train = load_train_data()
X_train /= 255

# for i in range(5):
#     display_image(X_train[i], y_train[i])

model = Sequential()

model.add(Conv2D(32, (3,3), input_shape=(image_width, image_height, 3), padding='same', activation='relu'))
model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
# model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=learning_rate), metrics=['accuracy'])

# datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

y_train = to_categorical(y_train)
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2, verbose=1)
# train_generator = datagen.flow(X_train, y_train, batch_size=batch_size, subset='training')
# validation_generator = datagen.flow(X_train, y_train, batch_size=batch_size, subset='validation')

# history = model.fit_generator(
#     train_generator,
#     validation_data=validation_generator,
#     steps_per_epoch=train_generator.n/batch_size,
#     validation_steps=validation_generator.n/batch_size,
#     epochs=epochs,
#     verbose=1)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validate'])
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validate'])
plt.show()

X_evaluate, y_evaluate = load_train_data(False)
X_evaluate /= 255

y_evaluate = to_categorical(y_evaluate)

evaluation = model.evaluate(X_evaluate, y_evaluate)
print('Model loss: %s\nModel accuracy: %s' % (evaluation[0], evaluation[1]))

# X_test, names = load_test_data()
# X_test /= 255

# predictions = model.predict(X_test)

# for i in range(5):
#     display_image(X_test[i], predictions[i][0])

# with open('dogs-vs-cats-results.csv', 'w') as f:
#     f.write('id,label\n')
#     for i in tqdm(range(predictions.shape[0])):
#         f.write('{},{}\n'.format(names[i], predictions[i][0]))
