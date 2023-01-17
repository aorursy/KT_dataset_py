import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import joblib
from tensorflow.keras.layers import Conv2D, Dense, Input, Dropout, GlobalMaxPooling2D, BatchNormalization, MaxPooling2D, Softmax
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD
from skimage import io
from skimage.transform import resize
from skimage.color import rgb2grey

!ls ../input/cats-vs-dogs
def create_model(data_shape):
    filters = 32
    model = Sequential()
    model.add(Input(shape=data_shape))
    for i in range(3):
        model.add(Conv2D(filters, (3,3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(filters, (3,3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        filters *= 2
    model.add(GlobalMaxPooling2D())
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(0.001, 0.9), loss='binary_crossentropy', metrics=['accuracy'])
    return model
def load_data(data_path):
    data = ImageDataGenerator(rescale=1.0/255.0)
    train_iter = data.flow_from_directory(data_path + '/train', class_mode='binary', batch_size=64, target_size=(200, 200))
    test_iter = data.flow_from_directory(data_path + '/test/test', class_mode='binary', batch_size=64, target_size=(200, 200))
    return (train_iter, test_iter)
train, test = load_data("../input/cats-vs-dogs/cats_n_dogs")
model = create_model((200, 200, 3))
result = model.fit(train, steps_per_epoch=len(train), epochs=20, validation_data=test, validation_steps=len(test))
model.evaluate(test)
model.save("C:\\Users\\G524366\\Documents\\AI_Course\\mymodel")
plt.plot(result.history['loss'], label='loss')
plt.plot(result.history['val_loss'], label='val_loss')
plt.legend()
plt.plot(result.history['accuracy'], label='accuracy')
plt.plot(result.history['val_accuracy'], label='val_accuracy')
plt.legend()
print(result.history)
predictions_gen = ImageDataGenerator(rescale=1.0/255.0)
predictions_iter = predictions_gen.flow_from_directory('../input/cats-vs-dogs/cats_n_dogs/predict',
                                                       class_mode='binary', batch_size=64, target_size=(200, 200))
probability_model = Sequential([model,Softmax()])
predictions = probability_model.predict(predictions_iter)
predictions
image = io.imread('../input/cats-vs-dogs/cats_n_dogs/test/test/cats/cat.4030.jpg', )
plt.imshow(image)
image = resize(image, (200, 200))
image = np.expand_dims(image, 0)
image.shape
predictions = model.predict(predictions_iter)
print(predictions)