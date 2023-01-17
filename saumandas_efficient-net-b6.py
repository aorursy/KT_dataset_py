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
        break
        

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_csv_2020_path = '/kaggle/input/jpeg-melanoma-256x256/train.csv'
test_df_path = '/kaggle/input/jpeg-melanoma-256x256/test.csv'
TRAIN_DIR = '/kaggle/input/jpeg-melanoma-256x256/train'
old_data_dir = '/kaggle/input/melanoma/dermmel/DermMel/train_sep'
train_df_full = pd.read_csv(train_csv_2020_path)
train_df = {}
train_df['image_name'] = [os.path.join(TRAIN_DIR, img_name+'.jpg') for img_name in train_df_full['image_name']]
train_df['target'] = train_df_full['target']
train_df = pd.DataFrame(train_df)
train_df.head()
for img in os.listdir(os.path.join(old_data_dir, 'Melanoma')):
    full_img = os.path.join(os.path.join(old_data_dir, 'Melanoma'), img)
    row_df = pd.DataFrame([[full_img, 1]], columns=['image_name', 'target'])
    train_df = train_df.append(row_df, ignore_index=True)
    
train_df.tail()
valid_dir_old = '/kaggle/input/melanoma/dermmel/DermMel/valid'
for img in os.listdir(os.path.join(valid_dir_old, 'Melanoma')):
    full_img = os.path.join(os.path.join(valid_dir_old, 'Melanoma'), img)
    row_df = pd.DataFrame([[full_img, 1]], columns=['image_name', 'target'])
    train_df = train_df.append(row_df, ignore_index=True)
test_dir_old = '/kaggle/input/melanoma/dermmel/DermMel/test'
for img in os.listdir(os.path.join(test_dir_old, 'Melanoma')):
    full_img = os.path.join(os.path.join(test_dir_old, 'Melanoma'), img)
    row_df = pd.DataFrame([[full_img, 1]], columns=['image_name', 'target'])
    train_df = train_df.append(row_df, ignore_index=True)
print(f'Melanoma instances: {sum(train_df.target)}')
print(f'Non-Melanoma instances: {len(train_df.target)-sum(train_df.target)}')
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler

print('Original dataset shape %s' % Counter(train_df['target']))
rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(np.array(train_df['image_name']).reshape(-1, 1), 
                                np.array(train_df['target']).reshape(-1, 1))
print('Original dataset shape %s' % Counter(y_res))

train = {}
train['image_name'] = X_res.reshape(X_res.shape[0])
train['target'] = y_res
train = pd.DataFrame(train)
train = train.sample(frac=1).reset_index(drop=True)
train.tail()
len(train)
import tensorflow as tf
from tensorflow import keras
train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale = 1./255, 
                                   horizontal_flip = True, 
                                   vertical_flip = True, 
                                   rotation_range = 45, 
                                   shear_range = 19,
                                   validation_split = 0.15)

train_generator = train_datagen.flow_from_dataframe(train,
                                                    x_col='image_name',
                                                    y_col='target',
                                                    target_size = (224, 224), 
                                                    class_mode = 'raw',
                                                    batch_size = 16,
                                                    shuffle = True,
                                                    subset = 'training')

val_generator = train_datagen.flow_from_dataframe(train,
                                                  x_col='image_name',
                                                  y_col='target',
                                                  target_size = (224, 224),
                                                  class_mode = 'raw',
                                                  batch_size = 8,
                                                  shuffle = True,
                                                  subset = 'validation')
!pip install -q efficientnet
from efficientnet.tfkeras import EfficientNetB6

model = keras.Sequential()
model.add(EfficientNetB6(input_shape=(224, 224, 3), include_top=False, weights='imagenet'))
model.add(keras.layers.GlobalAveragePooling2D())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid')) #binary output layer

model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.0001), metrics=['accuracy', 'AUC'])
history = model.fit(train_generator, epochs=10, steps_per_epoch=train_generator.n//train_generator.batch_size,
         validation_data=val_generator, validation_steps=val_generator.n//val_generator.batch_size)
test = pd.read_csv(test_df_path)
test_df = {}
test_df['image_name'] = [img+'.jpg' for img in test['image_name']]
test_df = pd.DataFrame(test_df)
test_df.head()
test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)

TEST_DIR = '/kaggle/input/jpeg-melanoma-256x256/test'
test_generator = test_datagen.flow_from_dataframe(test_df, TEST_DIR, batch_size=1, shuffle=False,
                                                 target_size=(224, 224),
                                                 x_col='image_name',
                                                 class_mode=None)
preds = model.predict_generator(test_generator)
preds
submission_path = '../input/jpeg-melanoma-256x256/sample_submission.csv'
sub = pd.read_csv(submission_path)
sub['target'] = preds
sub.head()
sub.to_csv('submission6.csv',index=False)

from IPython.display import FileLink
FileLink(r'submission6.csv')

model = keras.Sequential()
model.add(EfficientNetB6(input_shape=(224, 224, 3), include_top=False, weights='noisy-student'))
model.add(keras.layers.GlobalAveragePooling2D())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid')) #binary output layer

model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.0001), metrics=['accuracy', 'AUC'])
history2 = model.fit(train_generator, epochs=10, steps_per_epoch=train_generator.n//train_generator.batch_size,
         validation_data=val_generator, validation_steps=val_generator.n//val_generator.batch_size)
test_preds = model.predict_generator(test_generator)
sub2 = pd.read_csv(submission_path)
sub2['target'] = test_preds
sub2.head()
sub2.to_csv('submission_noisy.csv',index=False)
FileLink(r'submission_noisy.csv')
ensemble1 = {}
ensemble1['image_name'] = sub['image_name']
ensemble1['target'] = 0.5*sub['target'] + 0.5*sub2['target']
ensemble1 = pd.DataFrame(ensemble1)
ensemble1.head()
ensemble1.to_csv('ensemble1.csv',index=False)
FileLink(r'ensemble1.csv')
ensemble2 = {}
ensemble2['image_name'] = sub['image_name']
ensemble2['target'] = 0.6*sub['target'] + 0.4*sub2['target']
ensemble2 = pd.DataFrame(ensemble2)
ensemble2.head()
ensemble1.to_csv('ensemble2.csv',index=False)
FileLink(r'ensemble2.csv')
import matplotlib.pyplot as plt
test_img_name='ISIC_0434285.jpg'
full_path = os.path.join(TEST_DIR, test_img_name)
img = keras.preprocessing.image.load_img(full_path)
plt.imshow(img)
