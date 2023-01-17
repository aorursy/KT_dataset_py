import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator, load_img
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
import zipfile
import os
import re
# DATASET_ZIP = 'data/tamil_characters.zip'
IMAGE_SHAPE = (120, 120, 1)
# with zipfile.ZipFile(DATASET_ZIP, 'r') as z:
#   z.extractall('data')
# os.rename('data/shuffled', 'data/train')
TRAIN_PATH = '/kaggle/input/tamil-handwritten-characters-dataset/shuffled'
print('Number of images in the dataset: ', len(os.listdir(TRAIN_PATH)))
files = os.listdir(TRAIN_PATH)
target = []
for filename in os.listdir(TRAIN_PATH):
  substr = re.search('_(.+?)t', filename)
  if(substr):
    category = substr.group(1)
    target.append(category)

# Create a DataFrame
df = pd.DataFrame({
    'filename': files,
    'category': target
})
print('Number of unique characters: {}'.format(df['category'].unique()))
df.head()
for each in df['category'].unique():
  filename = df[df['category'] == each]['filename'].iloc[0]
  plt.figure()
  img = mpimg.imread(os.path.join(TRAIN_PATH, filename))
  plt.imshow(img)
  plt.title(filename)
  plt.show()
# Map Tamil Character Category values to its equivalent Unicode characters
MAP = {
    '000':u'\u0B85', 
    '001':u'\u0B86', 
    '002':u'\u0B87', 
    '003':u'\u0B88', 
    '004':u'\u0B89', 
    '005':u'\u0B8A', 
    '006':u'\u0B8E', 
    '007':u'\u0B8F', 
    '008':u'\u0B90', 
    '009':u'\u0B92', 
    '010':u'\u0B93', 
    '155':u'\u0B94'
    }
MAP.items()
df['category'].value_counts().plot.bar()
# Drop the class with less data
df.drop(df[df['category'] == '155'].index, inplace = True) 
df['category'].value_counts()
for filename in df[df['category']=='006']['filename']:
  plt.figure()
  img = mpimg.imread(os.path.join(TRAIN_PATH, filename))
  plt.imshow(img)
  plt.title(filename)
  plt.show()
# height, width, depth = im_data.shape
height, width, depth = [], [], []
for filename in df['filename']:
  img = mpimg.imread(os.path.join(TRAIN_PATH, filename))
  h, w, d = img.shape
  height.append(h)
  width.append(w)
  depth.append(d)

dim_df = pd.DataFrame({
    'height': height,
    'width': width,
    'depth': depth
})
dim_df.describe()
train_df, val_df = train_test_split(df, test_size=0.2, random_state=28) 
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
print('Train Dataset Size: ', len(train_df))
print('Validation Dataset Size: ', len(val_df))


train_df['category'].value_counts().plot.bar()
plt.title('Train Dataset Data Distribution')
plt.show()

plt.figure()

val_df['category'].value_counts().plot.bar()
plt.title('Validation Dataset Data Distribution')
plt.show()
batch_size = 5
epoch = 50

train_count = train_df.shape[0]
val_count = val_df.shape[0]

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    # rotation_range = 10,
    # width_shift_range = 0.2,
    # height_shift_range = 0.2,
    # shear_range = 0.2,
    horizontal_flip=False,
    fill_mode='nearest', 
    )

train_gen = train_datagen.flow_from_dataframe(
    train_df,
    directory = TRAIN_PATH,
    x_col = 'filename',
    y_col = 'category',
    class_mode = 'categorical',
    target_size = IMAGE_SHAPE[:2],
    batch_size = batch_size,
    color_mode='grayscale'
)

val_gen = train_datagen.flow_from_dataframe(
    val_df,
    directory = TRAIN_PATH,
    x_col = 'filename',
    y_col = 'category',
    class_mode = 'categorical',
    target_size = IMAGE_SHAPE[:2],
    batch_size = batch_size,
    color_mode='grayscale'
) 
def build_model():
  model = models.Sequential()

  model.add(layers.Conv2D(32, (5, 5), activation='relu', input_shape=IMAGE_SHAPE))
  model.add(layers.BatchNormalization())
  model.add(layers.MaxPool2D(pool_size=(2, 2)))
  model.add(layers.Dropout(0.2))

  model.add(layers.Conv2D(32, (5, 5), activation='relu'))
  model.add(layers.BatchNormalization())
  model.add(layers.MaxPool2D(pool_size=(2, 2)))
  model.add(layers.Dropout(0.2))

  model.add(layers.Conv2D(32, (5, 5), activation='relu'))
  model.add(layers.BatchNormalization())
  model.add(layers.MaxPool2D(pool_size=(2, 2)))
  model.add(layers.Dropout(0.2))

  model.add(layers.Conv2D(64, (6, 6), activation='relu'))
  model.add(layers.BatchNormalization())
  model.add(layers.MaxPool2D(pool_size=(2, 2)))
  model.add(layers.Dropout(0.2))
  
  model.add(layers.Flatten())
  model.add(layers.Dense(256, activation='relu'))
  model.add(layers.BatchNormalization())
  model.add(layers.Dense(11, activation='softmax'))

  model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
  return model

model = build_model()
model.summary()
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

earlystop = EarlyStopping(patience=10)
lrreducuction = ReduceLROnPlateau(monitor='val_loss', factor=0.05, patience=2, verbose=1, min_lr=0.000005)
filepath = "checkpoint.h5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')

callbacks = [earlystop, lrreducuction, checkpoint]
history = model.fit(
    train_gen,
    epochs=epoch,
    steps_per_epoch = train_count // batch_size,
    validation_data = val_gen,
    validation_steps = val_count // batch_size,
    callbacks = callbacks
)
model.save_weights("tamil_char.h5")
model.save("tamil_handwritten_char.h5")
epoch_xaxis = range(1, len(history.history['accuracy'])+1)

plt.plot(epoch_xaxis, history.history['accuracy'], 'r', label='Training Accuracy')
plt.plot(epoch_xaxis, history.history['val_accuracy'], 'b', label='Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.show()

plt.figure()

plt.plot(epoch_xaxis, history.history['loss'], 'r', label='Training Loss')
plt.plot(epoch_xaxis, history.history['val_loss'], 'b', label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.show()
sample_test = val_df.head(18)
sample_test.head()
plt.figure(figsize=(12, 24))
for index, row in sample_test.iterrows():
    filename = row['filename']
    category = row['category']
    img = load_img(os.path.join(TRAIN_PATH,filename), target_size=IMAGE_SHAPE)
    plt.subplot(6, 3, index+1)
    plt.imshow(img)
    plt.xlabel(filename + '(' + "{}".format(category) + ')' )
plt.tight_layout()
plt.show()


for index, row in sample_test.iterrows():
    filename = row['filename']
    category = row['category']
    print(filename + ' ==> ' + MAP[category])

