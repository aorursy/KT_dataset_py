import numpy as np
import pandas as pd
import tensorflow as tf
import os
BASE_DIR = "/kaggle/input/2ndparrot/"
train_dir = BASE_DIR + '2020_parrot_dataset_final/2020_parrot_dataset/train'
sample = pd.read_csv(BASE_DIR + 'submission_1.csv')
sample.head()
from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(rotation_range=15,
                            width_shift_range = 0.2,
                            height_shift_range = 0.2,
                            rescale = 1./255,
                            shear_range = 0.2,
                            zoom_range = 0.2,
                            horizontal_flip = False,
                            fill_mode = 'nearest',
                            validation_split = 0.2)

# 데이터셋 생성
batch_size = 128
np.random.seed(123457)
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size = (220, 220),              
    batch_size = batch_size,
    subset = 'training',
    class_mode = 'categorical', #one-hot
    shuffle = True)
valid_generator = datagen.flow_from_directory(
    train_dir,
    target_size = (220, 220),
    batch_size = batch_size,
    shuffle = True,
    subset = 'validation',
    class_mode = 'categorical'
)
def my_gen(gen):
    while True:
        try:
            data, labels = next(gen)
            yield data, labels
        except:
            pass

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten
from tensorflow.keras.layers import MaxPool2D, ZeroPadding2D
import tensorflow as tf
print(tf.__version__)
tf.test.is_gpu_available()
from tensorflow.python.client import device_lib
device_lib.list_local_devices()
# Baseline model
tf.keras.backend.clear_session()
N_CH = 3

model = Sequential()
model.add(Conv2D(32, (3, 3), activation = 'relu', padding = 'same',
                 input_shape = (220, 220, 3)))
model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(MaxPool2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding = 'same', activation = 'relu'))
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPool2D(2, 2))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(8, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy',
             optimizer = 'adam',
             metrics = ['accuracy'])
model.summary()
hist = model.fit_generator(my_gen(train_generator),
                           steps_per_epoch = 7811//batch_size,
                           epochs = 15,
                           validation_data = my_gen(valid_generator),
                           validation_steps = 1948//batch_size,
                           verbose = 1)
from tqdm.auto import tqdm
import cv2
target = []
row_id = []
x_test = []
test_dir = BASE_DIR + '2020_parrot_dataset_final/2020_parrot_dataset/test'
for img in tqdm(os.listdir(test_dir)):
    temp = cv2.imread(os.path.join(test_dir,img))
    row_id.append(os.path.basename(img))
#     if temp is not None:
    temp = cv2.resize(temp, (220, 220), interpolation = cv2.INTER_AREA)
#     else:
#         temp = np.zeros((220, 220, 3))
    x_test.append(temp)
x_test = np.array(x_test)
x_test = x_test/255
pred = model.predict(x_test)
pred = np.argmax(pred, axis = 1)
pred
df_sub = pd.DataFrame({'name' : row_id, 'answer' : pred}, dtype = 'U')
display(df_sub.head())
df_sub.to_csv("submission.csv", index = False)
