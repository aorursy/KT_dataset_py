# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from matplotlib.image import imread
os.listdir('/kaggle/input/cell-images-for-detecting-malaria/cell_images/cell_images/')
imread('/kaggle/input/cell-images-for-detecting-malaria/cell_images/Uninfected/C110P71ThinF_IMG_20150930_105559_cell_97.png').shape
img = imread('/kaggle/input/cell-images-for-detecting-malaria/cell_images/Parasitized/C137P98ThinF_IMG_20151005_161449_cell_5.png')
import matplotlib.pyplot as plt

plt.imshow(img)
from keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(rotation_range=20,

                                  width_shift_range=0.1,

                                  height_shift_range=0.1,

                                  shear_range=0.1,

                                  zoom_range=0.1,

                                  horizontal_flip=True,

                                  fill_mode='nearest',

                                  validation_split=0.33)
plt.imshow(train_datagen.random_transform(img))
train_dir='/kaggle/input/cell-images-for-detecting-malaria/cell_images/cell_images/'
train_gen = train_datagen.flow_from_directory(train_dir,

                                              target_size=(130, 130),

                                              batch_size=16,

                                              class_mode='binary',

                                              subset='training')



val_gen = train_datagen.flow_from_directory(train_dir,

                                              target_size=(130, 130),

                                              batch_size=16,

                                              class_mode='binary',

                                              subset='validation',

                                            shuffle=False)
from keras.models import Sequential

from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
model = Sequential()

model.add(Conv2D(32, (3,3), input_shape=(130,130,3), activation='relu'))

model.add(MaxPooling2D())

model.add(Conv2D(64, (3,3), activation='relu'))

model.add(MaxPooling2D())

model.add(Conv2D(64, (3,3), activation='relu'))

model.add(MaxPooling2D())



model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.2))



model.add(Dense(1, activation='sigmoid'))



model.compile(optimizer = 'adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
from keras.callbacks import EarlyStopping, ModelCheckpoint



early_stop = EarlyStopping(monitor='val_loss', patience=2)
train_gen.class_indices
checkpoint = ModelCheckpoint("best_model.hdf5", monitor='loss', verbose=1,save_best_only=True, mode='auto', period=1)

results = model.fit_generator(train_gen, epochs=20, validation_data=val_gen, callbacks=[early_stop, checkpoint])
import pandas as pd

metrics = pd.DataFrame(results.history)
metrics
metrics[['accuracy','val_accuracy']].plot()

metrics[['loss','val_loss']].plot()
model.evaluate_generator(val_gen)
model.metrics_names
pred_classes = model.predict_classes(val_gen)
pred_gen = model.predict_generator(val_gen)
pred = model.predict(val_gen)
pred_classes
pred_gen
val_gen.classes, val_gen.class_indices
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(val_gen.classes, pred_classes))
cm = confusion_matrix(val_gen.classes, pred_classes)

import seaborn as sns

sns.heatmap(cm, annot=True)
img = imread('/kaggle/input/cell-images-for-detecting-malaria/cell_images/Parasitized/C137P98ThinF_IMG_20151005_161449_cell_5.png')

plt.imshow(img)
from keras.preprocessing import image

my_image = image.load_img('/kaggle/input/cell-images-for-detecting-malaria/cell_images/Parasitized/C137P98ThinF_IMG_20151005_161449_cell_5.png', target_size=(130,130,3))
my_image
my_img_arr = image.img_to_array(my_image)

my_img_arr.shape
import numpy as np

my_img_arr = np.expand_dims(my_img_arr, axis=0)
my_img_arr.shape
model.predict_classes(my_img_arr)
train_gen.class_indices