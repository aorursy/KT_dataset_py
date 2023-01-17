import tensorflow as tf

from tensorflow.keras import layers #capas de la red
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping

import PIL
import pandas as pd #metricas
import numpy as np #argmax

import matplotlib.pyplot as plt 
%matplotlib inline 

data_path = '../input/chest-xray-covid19-pneumonia/Data'
test_data_path = '../input/chest-xray-covid19-pneumonia/Data/test'
train_data_path = '../input/chest-xray-covid19-pneumonia/Data/train'
import pathlib
path = pathlib.Path()
print(path)
print(type(path))

data_dir_train = pathlib.Path(train_data_path)  #saving directories 
print((data_dir_train))
data_dir_test = pathlib.Path(test_data_path)
print(type(data_dir_test))
print(data_dir_train.glob)

image_train_count = len(list(data_dir_train.glob('*/*.jpg')))  #image count
print("cardinality of train set=",image_train_count)
image_test_count = len(list(data_dir_test.glob('*/*.jpg')))
print("cardinality of test set=",image_test_count)
#print("total images=cardinality of the dataset=",image_train_count+image_test_count)

COVID19 = list(data_dir_train.glob('COVID19/*'))  #make a list with images whose name contains "COVID19"
PNEUMONIA = list(data_dir_train.glob('PNEUMONIA/*'))
NORMAL = list(data_dir_train.glob('NORMAL/*'))

PIL.Image.open(str(COVID19[5]))
#PIL.Image.open(str(NORMAL[0]))
#PIL.Image.open(str(PNEUMONIA[19]))

image_height = 128 #altura
image_width = 128 #ancho

image_shape = (image_height, image_width, 1)

image_gen = ImageDataGenerator(#rescale=1./255
                               )
num_classes = 3

model = Sequential([
  layers.experimental.preprocessing.Rescaling(1./255, input_shape=image_shape),
  layers.Conv2D(128, 3, padding='same', activation='relu'), 
  layers.MaxPooling2D(2),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(2),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(2),
  layers.Flatten(),
  layers.Dense(64, activation='sigmoid'),
  layers.Dropout((0.5)), 
  layers.Dense(num_classes, activation = 'softmax')
])



model.compile(optimizer='adam', #adam es la función de optimizazion recomendada en clase
              loss=tf.keras.losses.CategoricalCrossentropy(), 
              metrics=['accuracy', 'Precision']) #metrica para monstrar durante el entrenamiento
model.summary()

batch_size = 32
train_image_gen = image_gen.flow_from_directory(train_data_path,
                                               target_size=(image_height,image_width),
                                                color_mode='grayscale',
                                               batch_size=batch_size,
                                               class_mode='categorical')
test_image_gen = image_gen.flow_from_directory(test_data_path,
                                               target_size=(image_height,image_width),
                                               color_mode='grayscale',
                                               batch_size=batch_size, shuffle=False) 
print(train_image_gen.class_indices)
print(test_image_gen.class_indices)
results = model.fit(train_image_gen,epochs=15,
                              validation_data=test_image_gen)
resumen = pd.DataFrame(model.history.history)
resumen
resumen[['accuracy', 'precision']].plot()
resumen[['accuracy', 'loss']].plot()
model.evaluate(test_image_gen)
predictions = np.argmax(model.predict(test_image_gen), axis = -1)
predictions
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(test_image_gen.classes,predictions))
print(confusion_matrix(test_image_gen.classes,predictions))

