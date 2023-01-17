import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import *
from keras.models import * 
from keras.preprocessing import image
TRAIN_PATH = "../input/chest-xray-pneumonia/chest_xray/train"
VAL_PATH = "../input/chest-xray-pneumonia/chest_xray/val"
model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3), activation='relu', input_shape=(224,224,3)))
model.add(Conv2D(64,kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(64,kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128,kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64,activation = 'relu'))
model.add(Dropout(0.25))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss = keras.losses.binary_crossentropy, optimizer='adam', metrics=['accuracy'])
model.summary()
train_datagen = image.ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
)

test_dataset = image.ImageDataGenerator(rescale=1./255)
train_gen = train_datagen.flow_from_directory(
     "../input/covid-dataset/CovidDataset/Train",
    target_size = (224,224),
    batch_size = 32,
    class_mode = 'binary',
)
train_gen.class_indices
validation_gen = test_dataset.flow_from_directory(
    '../input/covid-dataset/CovidDataset/Val',
    target_size = (224,224),
    batch_size = 32,
    class_mode = 'binary',
)
validation_gen.class_indices
hist = model.fit_generator(
    train_gen,
    steps_per_epoch = 8,
    validation_data = validation_gen,
    validation_steps = 2,
    epochs = 10
)
model.save("model_adv.h5")
model.evaluate_generator(train_gen)
model.evaluate_generator(validation_gen)
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()
model = load_model('model_adv.h5')
import os
train_gen.class_indices
y_actual = []
y_test = []
for i in os.listdir('../input/covid-dataset/CovidDataset/Val/Normal'):
  img = image.load_img('../input/covid-dataset/CovidDataset/Val/Normal/'+i, target_size=(224,224))
  img = image.img_to_array(img)
  img = np.expand_dims(img, axis=0)
  p = model.predict_classes(img)
  y_test.append(p[0,0])
  y_actual.append(1)
for i in os.listdir('../input/covid-dataset/CovidDataset/Val/Covid'):
  img = image.load_img('../input/covid-dataset/CovidDataset/Val/Covid/'+i, target_size=(224,224))
  img = image.img_to_array(img)
  img = np.expand_dims(img, axis=0)
  p = model.predict_classes(img)
  y_test.append(p[0,0])
  y_actual.append(0)
y_actual = np.array(y_actual)
y_test = np.array(y_test)
y_actual, y_test
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_actual, y_test)
import seaborn as sns
sns.heatmap(cm, cmap="plasma", annot=True)
train_datagen = image.ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
)

test_dataset = image.ImageDataGenerator(rescale=1./255)
train_gen = train_datagen.flow_from_directory(
    '../input/covid-dataset/CovidDataset/Train',
    target_size = (224,224),
    batch_size = 32,
    class_mode = 'binary',
    subset='training'
)
train_gen.class_indices
ieee_gen = test_dataset.flow_from_directory(
    '../input/ieeeimagedataset/datasetcovid19/',
    target_size = (224,224),
    batch_size = 32,
    class_mode = 'binary',
)
ieee_gen.class_indices
hist = model.fit_generator(
    train_gen,
    steps_per_epoch = 6,
    validation_data = ieee_gen,
    validation_steps = 2,
    epochs = 5,
)
model.save("model_adv.h5")
model.evaluate_generator(train_gen)
model.evaluate_generator(ieee_gen)
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()
model = load_model('model_adv.h5')
import os
train_gen.class_indices
y_actual = []
y_test = []
for i in os.listdir('../input/ieeeimagedataset/datasetcovid19/Normal'):
  img = image.load_img('../input/ieeeimagedataset/datasetcovid19/Normal/'+i, target_size=(224,224))
  img = image.img_to_array(img)
  img = np.expand_dims(img, axis=0)
  p = model.predict_classes(img)
  y_test.append(p[0,0])
  y_actual.append(1)
for i in os.listdir('../input/ieeeimagedataset/datasetcovid19/Covid'):
  img = image.load_img('../input/ieeeimagedataset/datasetcovid19/Covid/'+i, target_size=(224,224))
  img = image.img_to_array(img)
  img = np.expand_dims(img, axis=0)
  p = model.predict_classes(img)
  y_test.append(p[0,0])
  y_actual.append(0)
y_actual = np.array(y_actual)
y_test = np.array(y_test)
y_actual, y_test
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_actual, y_test)
import seaborn as sns
sns.heatmap(cm, cmap="plasma", annot=True)
