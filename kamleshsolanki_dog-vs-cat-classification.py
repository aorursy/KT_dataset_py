from keras.preprocessing.image import ImageDataGenerator

from zipfile import ZipFile

import matplotlib.pyplot as plt

from keras.applications.inception_resnet_v2 import InceptionResNetV2

from keras.layers import Dense, Dropout, InputLayer

from keras import Sequential

from keras.callbacks import EarlyStopping, LearningRateScheduler
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames[:10]:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_dir = '/kaggle/input/dogs-vs-cats/train.zip'

os.mkdir('/kaggle/working/tmp')

with ZipFile(train_dir) as zipfile:

    zipfile.extractall('/kaggle/working/tmp')
filenames = []

class_name = []

for name in os.listdir('/kaggle/working/tmp/train'):

    filenames.append(name)

    class_name.append(name.split('.')[0])

train_df = pd.DataFrame(dict({'filename' : filenames, 'class' : class_name}))                    
train_df.head()
train_dataset = ImageDataGenerator(rescale = 1 / 255.0,

                                  rotation_range = 15,

                                  width_shift_range=0.1,

                                  height_shift_range=0.1,

                                  shear_range = 0.1,

                                  zoom_range = 0.1,

                                  horizontal_flip = True)

train_generator = train_dataset.flow_from_dataframe(train_df[:23000],

                                                  '/kaggle/working/tmp/train',

                                                   x_col = 'filename',

                                                   y_col = 'class',

                                                   batch_size = 128,

                                                   class_mode = 'categorical')



validation_dataset = ImageDataGenerator(rescale = 1 / 255.0,

                                  rotation_range = 15,

                                  width_shift_range=0.1,

                                  height_shift_range=0.1,

                                  shear_range = 0.1,

                                  zoom_range = 0.1,

                                  horizontal_flip = True)

validation_generator = train_dataset.flow_from_dataframe(train_df[23000:],

                                                  '/kaggle/working/tmp/train',

                                                   x_col = 'filename',

                                                   y_col = 'class',

                                                   batch_size = 128,

                                                   class_mode = 'categorical')
class_to_indices = train_generator.class_indices

indices_to_class ={class_to_indices[key]:key for key in class_to_indices.keys()}

plt.figure(figsize=(12, 12))

for i in range(0, 15):

    plt.subplot(3, 5, i+1)

    for X_batch, Y_batch in train_generator:

        image = X_batch[0]

        plt.xticks([])

        plt.yticks([])

        plt.xlabel(indices_to_class[np.argmax(Y_batch[0])])

        plt.imshow(image)

        break

plt.tight_layout()

plt.show()
base_model = InceptionResNetV2(input_shape = (256, 256, 3), include_top = False, pooling = 'avg')

base_model.trainable = False

model = Sequential()

model.add(base_model)

model.add(Dropout(0.7))

model.add(Dense(2, activation = 'softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.summary()
def scheduler(epoch):

    return 0.001 * np.exp(0.1 *  epoch)



learning_callback = LearningRateScheduler(scheduler)

early_stop_callback = EarlyStopping(monitor='val_loss', patience = 2)

my_callback = [learning_callback, early_stop_callback]
model.fit_generator(train_generator, epochs = 10, validation_data=validation_generator, callbacks = my_callback)
test_dir = '/kaggle/input/dogs-vs-cats/test1.zip'

with ZipFile(test_dir) as zipfile:

    zipfile.extractall('/kaggle/working/tmp')
filenames = []

class_name = []

for name in os.listdir('/kaggle/working/tmp/test1'):

    filenames.append(name)

    class_name.append(name.split('.')[0])

test_df = pd.DataFrame(dict({'filename' : filenames}))                      
test_dataset = ImageDataGenerator(rescale = 1 / 255.0)

test_generator = test_dataset.flow_from_dataframe(test_df,

                                                  '/kaggle/working/tmp/test1',

                                                   x_col = 'filename',

                                                   y_col = None,

                                                   batch_size = 1,

                                                   class_mode = None)
prediction = model.predict_generator(test_generator)

prediction = np.argmax(prediction, axis = 1)
submission = pd.DataFrame({'id' : test_generator.filenames, 'label' : prediction})

submission['id'] = submission['id'].apply(lambda x : int(x.split('.')[0]))

submission = submission.sort_values(by = 'id')
submission.to_csv('submission.csv', index = False)
submission.head()