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
import zipfile

with zipfile.ZipFile("../input/dogs-vs-cats/train.zip","r") as z:
    z.extractall(".")
    
with zipfile.ZipFile("../input/dogs-vs-cats/test1.zip","r") as z:
    z.extractall(".")
files_names = os.listdir("/kaggle/working/train")
dog_cat = []


for file in files_names:
    dog_or_cat = file.split('.')[0]
    if dog_or_cat == 'dog':
        dog_cat.append(1)
    else:
        dog_cat.append(0)
        
df = pd.DataFrame({'files_name': files_names, 'category': dog_cat })

df.head()
    
import matplotlib.pyplot as plt
import seaborn as sns
import random
from keras.preprocessing.image import ImageDataGenerator, load_img


sample = random.choice(files_names)

sample = load_img("/kaggle/working/train/" + sample)


plt.imshow(sample)



sns.countplot(df['category'])
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


IMAGE_WIDTH=128
IMAGE_HEIGHT=128
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3


df['category'] = df['category'].replace({0:'cat', 1:'dog'})
display(df.head())
model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT,IMAGE_CHANNELS)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


model.add(Conv2D(64, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


model.add(Conv2D(128, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(528, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
          
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


model.summary()

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

early_stopping = EarlyStopping(patience=8)

reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5,
                              patience=2, min_lr=0.00001, verbose=1)

train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)
total_train = train_df.shape[0]
total_validate = validate_df.shape[0]
batch_size=15
train_datagen = ImageDataGenerator(width_shift_range=0.1, 
                                   height_shift_range=0.1,
                                   brightness_range=(0.5, 2.5), 
                                   zoom_range=0.35, 
                                   horizontal_flip=True,
                                   rescale = 1./255,
                                   rotation_range=15,
                                   shear_range=0.1
                                   
                                  )

train_generator = train_datagen.flow_from_dataframe(train_df,"/kaggle/working/train",
                                                   x_col='files_name', y_col='category',
                                                   target_size=IMAGE_SIZE,
                                                   class_mode='categorical',
                                                   batch_size=15)



validation_datagen = ImageDataGenerator(rescale= 1./255)

valid_generator = validation_datagen.flow_from_dataframe(validate_df,"/kaggle/working/train",
                                                     x_col='files_name', y_col='category',
                                                     target_size=IMAGE_SIZE,
                                                     class_mode='categorical',
                                                     batch_size=15)
example = train_df.sample(n=1).reset_index(drop=True)

example_generator = train_datagen.flow_from_dataframe(example, "/kaggle/working/train", x_col='files_name', y_col='category',
                                                     target_size=IMAGE_SIZE,
                                                     class_mode='categorical'
                                                     )


plt.figure(figsize=(12,12))
for i in range (0,15):
    plt.subplot(5, 3, i+1)
    for X_batch, Y_batch in example_generator:
        image = X_batch[0]
        plt.imshow(image)
        break
plt.tight_layout()
plt.show()
epochs = 50
history = model.fit(train_generator, 
                    epochs=epochs, 
                    validation_data= valid_generator, 
                    validation_steps= total_validate/15, 
                    steps_per_epoch= total_train/15,
                    callbacks=[early_stopping, reduce_lr], verbose=1
                   
                   )

model.save_weights("model.h5")
test_filenames = os.listdir("/kaggle/working/test1")
test_df = pd.DataFrame({
    'filename': test_filenames
})
nb_samples = test_df.shape[0]


test_gen = ImageDataGenerator(rescale=1./255)
test_generator = test_gen.flow_from_dataframe(
    test_df, 
    "/kaggle/working/test1", 
    x_col='filename',
    y_col=None,
    class_mode=None,
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
    shuffle=False
)
predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples/15))
test_df['category'] = np.argmax(predict, axis=-1)
label_map = dict((v,k) for k,v in train_generator.class_indices.items())
test_df['category'] = test_df['category'].replace(label_map)
test_df['category'] = test_df['category'].replace({ 'dog': 1, 'cat': 0 })
sample_test = test_df.head(18)
sample_test.head()
plt.figure(figsize=(12, 24))
for index, row in sample_test.iterrows():
    filename = row['filename']
    category = row['category']
    img = load_img("/kaggle/working/test1/"+filename, target_size=IMAGE_SIZE)
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