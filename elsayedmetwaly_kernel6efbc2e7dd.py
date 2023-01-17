import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.applications import imagenet_utils

from keras.applications.mobilenet import MobileNet
from keras.models import Model
from PIL import Image
from keras.layers import Input, AveragePooling2D, GlobalAveragePooling2D, Concatenate
train_dir = '../input/chest-xray-covid19-pneumonia/Data/train'
valid_dir= '../input/chest-xray-covid19-pneumonia/Data/test'
test_dir = '../input/covid19-patient-xray-image-dataset/COVID-19 patient X-ray image dataset/corona/test'
baseModel = MobileNet(input_shape=(224,224,3), weights='imagenet', include_top=False)

for layer in baseModel.layers[:-4]:
  layer.trainable = False
model = Sequential()
model.add(baseModel)
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.5))
model.add(Dense(512))
model.add(Dropout(0.5))
model.add(Dense(3,activation = 'softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.summary()
train_datagen=ImageDataGenerator(rescale=1./255,
                          rotation_range= 10,
                          zoom_range=0.2,
                          horizontal_flip=True,
                          fill_mode='nearest',
                          brightness_range = [0.8, 1.2])

test_datagen= ImageDataGenerator(rescale=1./225)


train_generator= train_datagen.flow_from_directory(
        train_dir,
        target_size=(224,224),
        batch_size=32,
        class_mode="categorical")


valid_generator= test_datagen.flow_from_directory(
        valid_dir,
        target_size=(224,224),
        batch_size=32,
        class_mode="categorical")
from keras.callbacks import EarlyStopping
epochsWithOutImprovement = 4 
early_stopping = EarlyStopping(monitor='val_loss', patience=epochsWithOutImprovement, verbose=1)


history=model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=40,
        validation_data=valid_generator,
        validation_steps=len(valid_generator),
        callbacks=[early_stopping])
test_accuracy = model.evaluate_generator(valid_generator,steps=40)
print('Testing Accuracy with TEST SET: {:.2f}%'.format(test_accuracy[1] * 100))

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()
import cv2

img = cv2.imread("../input/chest-xray-covid19-pneumonia/Data/test/COVID19/COVID19(124).jpg")
print(img.shape)
img = cv2.resize(img,(224,224))
print(img.shape)

img = img.astype('float32')
img = img/255
img = np.expand_dims(img,axis=0)
print(img.shape)
pred = model.predict(img)
print(pred)
model.save('vgg16.h5')

model.save_weights('vgg16_weights.h5')

model_json = model.to_json()
with open('vgg16.json', 'w') as json_file:
    json_file.write(model_json)