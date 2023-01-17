

import os
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread


from keras import backend as K
from tensorflow.keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D
from keras.layers import Activation, Dropout, BatchNormalization, Flatten, Dense, AvgPool2D,MaxPool2D
from keras.models import Sequential, Model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.optimizers import Adam, SGD, RMSprop
%matplotlib inline
data = '../input/covid-19-x-ray-10000-images/dataset'
os.listdir(data)
test_path = my_data_dir+'/test/'
train_path = my_data_dir+'/train/'
img = imread(test_path+'IM-0025-0001.jpeg')
plt.imshow(img, cmap='gray')
img_width = 150
img_ht = 150
ch =3

image_shape = (img_width, img_ht, ch)
epochs = 48
batch_size = 16
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=image_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(Conv2D(250,(3,3)))
model.add(Activation("relu"))
  
model.add(Conv2D(128,(3,3)))
model.add(Activation("relu"))
model.add(AvgPool2D(2,2))
model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(AvgPool2D(2,2))

model.add(Conv2D(256,(2,2)))
model.add(Activation("relu"))
model.add(MaxPool2D(2,2))
    
model.add(Flatten())
model.add(Dense(32))
model.add(Dropout(0.50))
model.add(Dense(1))
model.add(Activation("sigmoid"))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss',patience=10)

image_gen = ImageDataGenerator(rotation_range=20, # rotate the image 20 degrees
                               width_shift_range=0.10, # Shift the pic width by a max of 5%
                               height_shift_range=0.10, # Shift the pic height by a max of 5%
                               rescale=1/255, # Rescale the image by normalzing it.
                               shear_range=0.1, # Shear means cutting away part of the image (max 10%)
                               zoom_range=0.1, # Zoom in by 10% max
                               horizontal_flip=True, # Allo horizontal flipping
                               fill_mode='nearest', # Fill in missing pixels with the nearest filled value
                               validation_split=0.3
                              )
train_image_gen = image_gen.flow_from_directory(data,
                                               target_size=image_shape[:2],
                                                color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='binary',
                                               subset='training')
test_image_gen = image_gen.flow_from_directory(data,
                                               target_size=image_shape[:2],
                                               color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='binary',shuffle=False,
                                               subset='validation')
train_image_gen.class_indices
import warnings
warnings.filterwarnings('ignore')

results = model.fit_generator(train_image_gen,epochs=epochs,
                              validation_data=test_image_gen,
                             callbacks=[early_stop])
losses = pd.DataFrame(model.history.history)
losses[['loss','val_loss']].plot()
model.metrics_names
model.evaluate_generator(test_image_gen)
pred_probabilities = model.predict_generator(test_image_gen)
pred_probabilities
test_image_gen.classes
predictions = pred_probabilities > 0.5
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(test_image_gen.classes,predictions))
confusion_matrix(test_image_gen.classes,predictions)
my_covid_image = image.load_img('../input/covin19-chest-infected-xray/chest_resize_md.png',target_size=image_shape)
my_covid_image

my_covid_image = image.img_to_array(my_covid_image)
# type(my_covid_image)
my_covid_image.shape
my_covid_image = np.expand_dims(my_covid_image, axis=0)

my_covid_image.shape
model.predict(my_covid_image)
train_image_gen.class_indices
