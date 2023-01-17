import pandas as pd
#custom function that returns only datasets with single playing card in it

def get_unique_classes(csv_file):

    df = pd.read_csv(csv_file)

    

    df_unique_series = df.groupby('filename')['filename'].count() <= 1

    

    unique_files = []

    for filename, isUnique in zip(df_unique_series.index, df_unique_series):

        if(isUnique):

            unique_files.append(filename)

            

    unique_set = []

    for idx, row in df.iterrows():

        if(row['filename'] in unique_files):

            unique_set.append((row['filename'], row['class']))

            

    return unique_set
unique_train_set = get_unique_classes('/kaggle/input/playing-card/card_dataset/train_labels.csv')

unique_test_set = get_unique_classes('/kaggle/input/playing-card/card_dataset/test_labels.csv')



print('unique_train_set', len(unique_train_set))

print('unique_test_set', len(unique_test_set))
from shutil import copyfile

import os

os.mkdir('/kaggle/cleaned_data')

os.mkdir('/kaggle/cleaned_data/train')

os.mkdir('/kaggle/cleaned_data/test')
src_dir = '/kaggle/input/playing-card/card_dataset/train/'

dst_dir = '/kaggle/cleaned_data/train'



for f, c in unique_train_set:

    src = os.path.join(src_dir, f)

    dst = os.path.join(dst_dir, c, f)

    if(os.path.exists(os.path.join(dst_dir, c)) != True):

        os.mkdir(os.path.join(dst_dir, c))

    copyfile(src, dst)

    

src_dir = '/kaggle/input/playing-card/card_dataset/test'

dst_dir = '/kaggle/cleaned_data/test'



for f, c in unique_test_set:

    src = os.path.join(src_dir, f)

    dst = os.path.join(dst_dir, c, f)

    if(os.path.exists(os.path.join(dst_dir, c)) != True):

        os.mkdir(os.path.join(dst_dir, c))

    copyfile(src, dst)
for dirname, _, filenames in os.walk('/kaggle/cleaned_data/'):

    if(len(filenames) != 0):

        print(os.path.basename(dirname), ' ==> ', len(filenames))

    else:

        print(dirname)
from keras_preprocessing import image
imgGen = image.ImageDataGenerator(rescale=1/255.)

train_datagen = imgGen.flow_from_directory('/kaggle/cleaned_data/train',

                                           target_size=(150,150),

                                           batch_size=10,

                                           )

val_datagen = imgGen.flow_from_directory('/kaggle/cleaned_data/test',

                                           target_size=(150,150),

                                           batch_size=10,

                                           )
from keras import models

from keras import layers

from keras import optimizers

from keras import regularizers
model = models.Sequential()

model.add(layers.Conv2D(32,(3,3),activation='relu', input_shape=(150,150,3)))

model.add(layers.MaxPooling2D())

model.add(layers.Conv2D(64,(3,3),activation='relu'))

model.add(layers.MaxPooling2D())

model.add(layers.Conv2D(64,(3,3),activation='relu'))

model.add(layers.MaxPooling2D())

model.add(layers.Conv2D(64,(3,3),activation='relu'))

model.add(layers.MaxPooling2D())

model.add(layers.Flatten())

model.add(layers.Dense(64,activation='relu'))

model.add(layers.Dense(6, activation='sigmoid'))
model.compile(optimizer=optimizers.RMSprop(lr=0.0006),loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit_generator(train_datagen, epochs=50,

                    steps_per_epoch = 18,

                    validation_data = val_datagen,

                    validation_steps = 3)
def get_smoothed(samples, factor = 0.9):

    smoothed = []

    for sample in samples:

        if smoothed:

            prev = smoothed[-1]

            smoothed.append((prev * factor) + (sample * (1-factor)))

        else:

            smoothed.append(sample)

    return smoothed
from matplotlib import pyplot as plt
plt.style.use('ggplot')

plt.plot(range(len(history.history['acc'])), get_smoothed(history.history['acc']), 'b', label='Training Accuracy')

plt.plot(range(len(history.history['acc'])), get_smoothed(history.history['val_acc']), 'bo', label='Validation Accuracy')

plt.legend()
model = models.Sequential()

model.add(layers.Conv2D(32,(3,3),activation='relu', input_shape=(150,150,3)))

model.add(layers.MaxPooling2D())

model.add(layers.Dropout(0.4))

model.add(layers.Conv2D(64,(3,3),activation='relu'))

model.add(layers.MaxPooling2D())

model.add(layers.Conv2D(64,(3,3),activation='relu'))

model.add(layers.MaxPooling2D())

model.add(layers.Conv2D(64,(3,3),activation='relu'))

model.add(layers.MaxPooling2D())

model.add(layers.Flatten())

model.add(layers.Dropout(0.4))

model.add(layers.Dense(64,activation='relu'))

model.add(layers.Dropout(0.3))

model.add(layers.Dense(6, activation='sigmoid'))
imgTrainGen = image.ImageDataGenerator(rescale=1/255., rotation_range=40,zoom_range = 0.2, 

                                  horizontal_flip=True, vertical_flip = True, 

                                  shear_range=0.2, height_shift_range=0.2, 

                                  width_shift_range=0.2)



train_datagen = imgTrainGen.flow_from_directory('/kaggle/cleaned_data/train',

                                           target_size=(150,150),

                                           batch_size=10,

                                           )
model.compile(optimizer=optimizers.RMSprop(lr=0.0006),loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit_generator(train_datagen, epochs=50,

                    steps_per_epoch = 18,

                    validation_data = val_datagen,

                    validation_steps = 3)
plt.plot(range(len(history.history['acc'])), get_smoothed(history.history['acc']), 'b', label='Training Accuracy')

plt.plot(range(len(history.history['acc'])), get_smoothed(history.history['val_acc']), 'bo', label='Validation Accuracy')

plt.legend()
history = model.fit_generator(train_datagen, epochs=50,

                    steps_per_epoch = 18,

                    validation_data = val_datagen,

                    validation_steps = 3)
plt.plot(range(len(history.history['acc'])), get_smoothed(history.history['acc']), 'b', label='Training Accuracy')

plt.plot(range(len(history.history['acc'])), get_smoothed(history.history['val_acc']), 'bo', label='Validation Accuracy')

plt.legend()