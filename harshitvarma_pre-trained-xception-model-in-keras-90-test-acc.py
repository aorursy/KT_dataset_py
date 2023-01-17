# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# DATASET PATHS:
## {O:'NORMAL', 1:'PNEUMONIA'} Mapping used in naming variable

BASE_PATH = '/kaggle/input/chest-xray-pneumonia/chest_xray/'

TRAIN_PATH = BASE_PATH + 'train/'
TEST_PATH  = BASE_PATH + 'test/'
VAL_PATH   = BASE_PATH + 'val/'

TRAIN_0 = TRAIN_PATH + 'NORMAL/'
TRAIN_1 = TRAIN_PATH + 'PNEUMONIA/'

TEST_0 = TEST_PATH + 'NORMAL/'
TEST_1 = TEST_PATH + 'PNEUMONIA/'

VAL_0 = VAL_PATH + 'NORMAL/'
VAL_1 = VAL_PATH + 'PNEUMONIA/'
print('The number of training images belonging to class 0: ', len(os.listdir(TRAIN_0)))
print('The number of training images belonging to class 1: ', len(os.listdir(TRAIN_1)))

print('The number of validation images belonging to class 0: ', len(os.listdir(VAL_0)))
print('The number of validation images belonging to class 1: ', len(os.listdir(VAL_1)))

print('The number of test images belonging to class 0: ', len(os.listdir(TEST_0)))
print('The number of test images belonging to class 1: ', len(os.listdir(TEST_1)))
from keras.preprocessing import image
filenames = os.listdir(TRAIN_0)
index = np.random.randint(0, len(filenames))
IMAGE_PATH = TRAIN_0 + filenames[index]

img = image.load_img(IMAGE_PATH)
img = image.img_to_array(img)/255.

%matplotlib inline
from matplotlib import pyplot as plt

plt.imshow(img)
plt.axis('off')
plt.title('Normal')
plt.show()
img.shape
def toDataframe(D_0, D_1):
    filenames_0 = np.array(os.listdir(D_0))
    y_0 = np.full(len(filenames_0), 'normal')

    filenames_1 = np.array(os.listdir(D_1))
    y_1 = np.full(len(filenames_1), 'pneumonia')

    filenames = np.append(filenames_0, filenames_1)
    y = np.append(y_0, y_1)

    df = pd.DataFrame({'Filenames':filenames, 'Classes':y})
    
    # Shuffling the data:
    df = df.sample(frac=1).reset_index(drop=True)
    
    return df
df_train = toDataframe(TRAIN_0, TRAIN_1)
df_val   = toDataframe(VAL_0, VAL_1)
df_test  = toDataframe(TEST_0, TEST_1)
display(df_train, df_val, df_test)
from keras.applications import Xception
# As of now, I have kept the input shape as (150, 150, 3)
model_base = Xception(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, BatchNormalization

#model_base.trainable = False

model = Sequential()
model.add(model_base) # Adding the base as a layer
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

model.summary()
from keras.preprocessing.image import ImageDataGenerator as IDG

train_datagen = IDG(
    rescale=1./255,
    rotation_range=5,
    width_shift_range=0.2,
    height_shift_range=0.2,
    #shear_range=0.2, #extra
    #zoom_range=0.2, #extra
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = IDG(rescale=1./255) # Don't augment these images

train_gen = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

validation_gen = test_datagen.flow_from_directory(
    VAL_PATH,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['acc']
)

m = len(df_train)
history = model.fit_generator(
    train_gen,
    steps_per_epoch=m/32.,
    epochs=10,
    validation_data=validation_gen,
    validation_steps=1
)
test_gen = test_datagen.flow_from_directory(
    TEST_PATH,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

m_test = len(df_test)
test_loss, test_acc = model.evaluate_generator(test_gen, steps=m_test/32)
model.save('xception_1.h5')
print('Test accuracy: ' ,test_acc)
