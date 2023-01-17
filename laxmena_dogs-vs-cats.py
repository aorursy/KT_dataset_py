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
from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras import optimizers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
import zipfile
from math import ceil

INPUT_PATH = '/kaggle/input/dogs-vs-cats'
WORKING_PATH = '/kaggle/working'


# Extract train and test zip files
with zipfile.ZipFile(os.path.join(INPUT_PATH, 'train.zip')) as z:
    z.extractall('.')
    
with zipfile.ZipFile(os.path.join(INPUT_PATH, 'test1.zip')) as z:
    z.extractall('.')
IMAGE_SHAPE = (150, 150, 3)
TRAIN_PATH = os.path.join(WORKING_PATH, 'train')
TEST_PATH = os.path.join(WORKING_PATH, 'test1')
files = os.listdir(TRAIN_PATH)
categories = []

for file in files:
    category = file.split('.')[0]
    if category == 'dog':
        categories.append(1)
    else:
        categories.append(0)

df = pd.DataFrame({
    'filename': files, 
    'category': categories
})

print(df.size)
df.head()
df['category'].value_counts().plot(kind='bar')
def build_model():
    model = models.Sequential()
    
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=IMAGE_SHAPE))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.20))
    
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))
    
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.30))
    
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.4))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(2, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model
model = build_model()
model.summary()
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
earlystop = EarlyStopping(patience=10)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                           patience=2,
                                           verbose=1,
                                           factor=0.5,
                                           min_lr=0.00001)
filepath = "model.h5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')

callbacks = [earlystop, learning_rate_reduction, checkpoint]
df['category'] = df['category'].replace({1:'dog', 0:'cat'})
train_df, valid_df = train_test_split(df, test_size=0.2, random_state=28)
train_df = train_df.reset_index(drop=True)
valid_df = valid_df.reset_index(drop=True)
train_df['category'].value_counts().plot.bar()
plt.title('Data Distribution - Training Dataset')
plt.show()
valid_df['category'].value_counts().plot.bar()
plt.title('Data Distribution - Validation Dataset')
plt.show()
train_count = len(train_df)
valid_count = len(valid_df)
batch_size = 25
train_datagen = ImageDataGenerator(rotation_range=20, 
                               rescale=1./255, 
                               width_shift_range=0.2, 
                               height_shift_range=0.2, 
                               shear_range=0.2,
                               horizontal_flip=True,
                               fill_mode='nearest')

train_gen = train_datagen.flow_from_dataframe(
    train_df,
    directory = TRAIN_PATH,
    x_col = 'filename',
    y_col = 'category',
    target_size = IMAGE_SHAPE[:2],
    class_mode = 'categorical',
    batch_size = batch_size
)

valid_gen = train_datagen.flow_from_dataframe(
    valid_df,
    directory = TRAIN_PATH,
    x_col = 'filename',
    y_col = 'category',
    target_size = IMAGE_SHAPE[:2],
    class_mode = 'categorical',
    batch_size = batch_size
)
epoch = 30
history = model.fit(
    train_gen,
    epochs=epoch,
    steps_per_epoch = train_count // batch_size,
    validation_data = valid_gen,
    
    validation_steps = valid_count // batch_size,
    callbacks = callbacks
)
model.save_weights("dog_vs_cat_3.h5")
epoch_count = range(1, len(history.history['accuracy'])+1)

plt.plot(epoch_count, history.history['accuracy'], 'bo', label='Training Accuracy')
plt.plot(epoch_count, history.history['val_accuracy'], 'b', label='Validation Accuracy')
plt.title('Accuracy')
plt.legend()
plt.show()

plt.figure()

plt.plot(epoch_count, history.history['loss'], 'bo', label='Training Loss')
plt.plot(epoch_count, history.history['val_loss'], 'r', label='Validation Loss')
plt.legend()
plt.show()
test_files = os.listdir(TEST_PATH)

test_df = pd.DataFrame({
    'filename': test_files
})

test_samples = test_df.shape[0]
test_gen = ImageDataGenerator(rescale=(1./255))
test_generator = test_gen.flow_from_dataframe(
    test_df,
    shuffle = False,
    directory = TEST_PATH,
    x_col = 'filename',
    y_col = None,
    target_size=IMAGE_SHAPE[:2],
    class_mode = None,
    batch_size = batch_size
)
from math import ceil
pred = model.predict(test_generator, steps=ceil(test_samples/batch_size))
test_df['category'] = np.argmax(pred, axis=-1)
label_map = dict((v,k) for k,v in train_gen.class_indices.items())
test_df['category'] = test_df['category'].replace(label_map)
test_df['category'] = test_df['category'].replace({ 'dog': 1, 'cat': 0 })
test_df['category'].value_counts().plot.bar()
sample_test = test_df.head(18)
sample_test.head()
plt.figure(figsize=(12, 24))
for index, row in sample_test.iterrows():
    filename = row['filename']
    category = row['category']
    img = load_img(os.path.join(TEST_PATH,filename), target_size=IMAGE_SHAPE)
    plt.subplot(6, 3, index+1)
    plt.imshow(img)
    plt.xlabel(filename + '(' + "{}".format(category) + ')' )
plt.tight_layout()
plt.show()
submission_df = test_df.copy()
submission_df['id'] = submission_df['filename'].str.split('.').str[0]
submission_df['label'] = submission_df['category']
submission_df.drop(['filename', 'category'], axis=1, inplace=True)
submission_df.to_csv('submission.csv', index=False)
