# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import zipfile

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
local_zip_train_path = '../input/dogs-vs-cats/train.zip'

zip_train_file = zipfile.ZipFile(local_zip_train_path, 'r')

zip_train_file.extractall('../kaggle/working')

zip_train_file.close()
local_zip_test_path = '../input/dogs-vs-cats/test1.zip'

zip_test_file = zipfile.ZipFile(local_zip_test_path, 'r')

zip_test_file.extractall('../kaggle/working')

zip_test_file.close()
print(os.listdir('../kaggle/working'))
filenames = os.listdir('../kaggle/working/train')



class_categ = []



for file in filenames:

    

    categ_str = file.split('.')[0]

    

    if categ_str == 'dog':

        class_categ.append('cat')

    else:

        class_categ.append('dog')



data_frame = pd.DataFrame({'file_name': filenames, 'category': class_categ})

data_frame
from sklearn.model_selection import train_test_split



df_train, df_validate =  train_test_split(data_frame, test_size = 0.2, random_state = 0)



df_train = df_train.reset_index(drop=True)

df_validate = df_validate.reset_index(drop=True)



training_data_size = df_train.shape[0]

validation_data_size = df_validate.shape[0]
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPool2D, Dropout, Flatten, Dense

model = tf.keras.models.Sequential()



model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())



model.add(Dense(512, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))



model.add(Dense(1024, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))



model.add(Dense(1, activation='sigmoid'))



model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(), loss= 'binary_crossentropy', metrics= ['accuracy'])
from tensorflow.keras.preprocessing.image import ImageDataGenerator



train_datagen =  ImageDataGenerator(rescale=1./255, 

                                    rotation_range=15,

                                    shear_range=0.1,

                                    zoom_range=0.2,

                                    horizontal_flip=True,

                                    width_shift_range=0.1,

                                    height_shift_range=0.1)



validate_datagen = ImageDataGenerator(rescale=1./255)



train_gen = train_datagen.flow_from_dataframe(df_train,  '../kaggle/working/train',x_col='file_name', y_col='category', batch_size= 64,target_size=(128,128), class_mode='binary')

val_gen = validate_datagen.flow_from_dataframe(df_validate,  '../kaggle/working/train',x_col='file_name', y_col='category', batch_size= 32,target_size=(128,128), class_mode='binary')
total_train = df_train.shape[0]

total_validate = df_validate.shape[0]

batch_size = 256

EPOCHS_ = 15
model_history = model.fit_generator(train_gen, 

                                    validation_data=val_gen, 

                                    epochs=EPOCHS_, 

                                    steps_per_epoch=total_train // batch_size, 

                                    validation_steps=total_validate // batch_size)
model.save_weights('model.h5')
from matplotlib import pyplot as plt



fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

ax1.plot(model_history.history['loss'], color='b', label="Training loss")

ax1.plot(model_history.history['val_loss'], color='r', label="validation loss")

ax1.set_xticks(np.arange(1, 15, 1))

ax1.set_yticks(np.arange(0, 1, 0.1))



ax2.plot(model_history.history['accuracy'], color='b', label="Training accuracy")

ax2.plot(model_history.history['val_accuracy'], color='r',label="Validation accuracy")

ax2.set_xticks(np.arange(1, 15, 1))



legend = plt.legend(loc='best', shadow=True)

plt.tight_layout()

plt.show()
test_filenames = os.listdir("../kaggle/working/test1")

test_df = pd.DataFrame({

    'filename': test_filenames

})

nb_samples = test_df.shape[0]
test_gen = ImageDataGenerator(rescale=1./255)

test_generator = test_gen.flow_from_dataframe(

    test_df, 

    "../kaggle/working/test1/", 

    x_col='filename',

    y_col=None,

    class_mode=None,

    target_size=(128,128),

    batch_size=batch_size,

    shuffle=False

)
predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples/batch_size))

predict
test_df['category'] = predict >= 0.5

test_df['category']
test_df['category'].value_counts().plot.bar()
submission_df = test_df.copy()

submission_df['id'] = submission_df['filename'].str.split('.').str[0]

submission_df['label'] = submission_df['category']

submission_df.drop(['filename', 'category'], axis=1, inplace=True)

submission_df.to_csv('submission.csv', index=False)