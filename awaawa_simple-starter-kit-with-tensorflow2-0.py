!unzip /kaggle/input/aerial-cactus-identification/train.zip

!unzip /kaggle/input/aerial-cactus-identification/test.zip
import os

import sys

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import matplotlib.image as pimg

import seaborn as sns

import math

from tqdm import tqdm

from PIL import Image



from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow.keras.preprocessing.image import load_img, ImageDataGenerator

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, GlobalMaxPooling2D

from tensorflow.keras import optimizers, regularizers

from tensorflow.keras.callbacks import Callback, EarlyStopping, LearningRateScheduler
print(sys.version)

print('tensorflow -> ', tf.__version__)
# When implementing machine learning code in TensorFlow, 

# TensorFlow often uses pseudo-random seed for things like weight initialization.

# As a result, the results change each time the code is re-executed, 

# and it's impossible to tell whether the result is due to a change in data or parameters or a random seed.

# Therefore, The random seed needs to be fixed.



np.random.seed(12)

tf.random.set_seed(12)
main_df = pd.read_csv('/kaggle/input/aerial-cactus-identification/train.csv')

sub_df = pd.read_csv('/kaggle/input/aerial-cactus-identification/sample_submission.csv')



train_dir = '/kaggle/working/train/'

test_dir = '/kaggle/working/test/'
main_df.head()
print('shape: ', main_df.shape)

print('===================================')

print(main_df['has_cactus'].value_counts())
plt.style.use('default')

sns.set()

sns.set_style('whitegrid')

sns.set_palette('Pastel2')



x = ['has cactus', 'hasn\'t cactus']

y = main_df.groupby('has_cactus').size()



fig = plt.figure()

ax = fig.add_subplot(1, 1, 1)

ax.pie(y, labels=x, autopct="%1.1f%%")



plt.show()
fig, ax = plt.subplots(2, 5, figsize = (12,6))



for i, idx in enumerate(main_df[main_df['has_cactus'] == 1]['id'][-5:]):

    path = os.path.join(train_dir, idx)

    img = load_img(path)

    ax[0, i].axis('off')

    ax[0, i].set_title('has cactus')

    ax[0, i].imshow(img)

    

for i, idx in enumerate(main_df[main_df['has_cactus'] == 0]['id'][-5:]):

    path = os.path.join(train_dir, idx)

    img = load_img(path)

    ax[1, i].axis('off')

    ax[1, i].set_title('hasn\'t cactus')

    ax[1, i].imshow(img)
train_df, val_df = train_test_split(main_df, test_size=0.25, stratify=main_df['has_cactus'], shuffle=True, random_state=12)



train_df = train_df.reset_index()

val_df = val_df.reset_index()



total_train = train_df.shape[0]

total_val = val_df.shape[0]



print('total_train: {}, total_val: {}'.format(total_train, total_val))
img_width, img_height = 32, 32

target_size = (img_width, img_height)



# Define Data Augmentation

train_datagen = ImageDataGenerator(rescale=1./255)

val_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)





# Convert the data type of 'has_cactus' to str to allow the model to be trained.

train_df['has_cactus'] = train_df['has_cactus'].astype(str)

val_df['has_cactus'] = val_df['has_cactus'].astype(str)
batch_size = 32

x_col, y_col = 'id', 'has_cactus'

class_mode = 'binary'





train_gen = train_datagen.flow_from_dataframe(train_df,

                                            train_dir,

                                            x_col=x_col,

                                            y_col=y_col,

                                            class_mode=class_mode,

                                            target_size=target_size,

                                            batch_size=batch_size,

                                            )



val_gen = val_datagen.flow_from_dataframe(val_df,

                                        train_dir,

                                        x_col=x_col,

                                        y_col=y_col,

                                        class_mode=class_mode,

                                        target_size=target_size,

                                        batch_size=batch_size,

                                        )
input_shape = (img_width, img_height, 3)

optimizer = optimizers.Adam(lr=1e-3)
model = Sequential()



model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu', input_shape=input_shape))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(64, kernel_size=(3,3), padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(128, kernel_size=(3,3), padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))



model.add(GlobalMaxPooling2D())



model.add(Dense(128, activation='relu'))

model.add(Dropout(0.25))



model.add(Dense(1, activation='sigmoid'))





model.compile(loss='binary_crossentropy', metrics=['acc'], optimizer=optimizer)

model.summary()
# Vary the learning rate according to the number of epochs.

def step_decay(epoch):

    initial_rate = 0.001

    drop = 0.5

    epochs_drop = 10.0

    lrate = initial_rate * math.pow(drop, math.floor((epoch) / epochs_drop))

    

    return lrate
lrate = LearningRateScheduler(step_decay)

es = EarlyStopping(monitor='val_loss', min_delta=0, patience=5)



callbacks = [lrate, es]
epochs = 30



history = model.fit(

    train_gen,

    epochs=epochs,

    steps_per_epoch=total_train//batch_size,

    validation_data=val_gen,

    validation_steps=total_val//batch_size,

    callbacks=callbacks,

    )
sns.set_palette('Dark2')

fig,ax = plt.subplots(2, 1)



plot_acc = pd.DataFrame({'acc': history.history['acc'],

                         'val_acc': history.history['val_acc']})



plot_loss = pd.DataFrame({'loss': history.history['loss'],

                          'val_loss': history.history['val_loss']})



plot_acc.plot(ax=ax[0])

plot_loss.plot(ax=ax[1])
def predict(model, sub_df):

    pred = np.empty((sub_df.shape[0],))

    

    for n in tqdm(range(sub_df.shape[0])):

        image = np.array(Image.open(test_dir + sub_df.id[n]))

        pred[n] = model.predict(image.reshape((1, 32, 32, 3))/255.0)[0]

    

    sub_df['has_cactus'] = pred

    return sub_df
predictions = predict(model, sub_df)
# If you ignore this process, you can't submit file.

# Maybe it's because there's more than just a file to submit in the working directory.

# Please let me know if you have any other solution to this.



!rm -r *
predictions.to_csv('submission.csv', header=True, index=False)