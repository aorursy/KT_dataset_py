import numpy as np

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os,cv2

from IPython.display import Image

from tqdm import tqdm, tqdm_notebook

from keras.preprocessing import image

from keras import optimizers

from keras import layers

from keras.models import Sequential

from keras.optimizers import Adam

from keras.applications.imagenet_utils import preprocess_input

from keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

from keras import regularizers

from keras.preprocessing.image import ImageDataGenerator

print(os.listdir("../input"))
train_dir="../input/train/train"

test_dir="../input/test/test"

train=pd.read_csv('../input/train.csv')



df_test=pd.read_csv('../input/sample_submission.csv')

train.has_cactus=train.has_cactus.astype(str)
train.tail()
train['has_cactus'].value_counts()

train_datagen = ImageDataGenerator(featurewise_center=False, samplewise_center=False, 

                             featurewise_std_normalization=False, 

                             samplewise_std_normalization=False, zca_whitening=False, zca_epsilon=1e-06, 

                             rotation_range= 55, 

                             width_shift_range=0.2, 

                             height_shift_range=0.2, 

                             brightness_range=None, 

                             shear_range=0.0, zoom_range=0.0, 

                             channel_shift_range=0.0, 

                             fill_mode='nearest',

                             cval=0.0, 

                             horizontal_flip=True, 

                             vertical_flip=False, 

                             rescale=1./255 , 

                             preprocessing_function=None, data_format=None, validation_split=0.0, dtype=None)
valid_datagen = ImageDataGenerator(rescale=1./255)
batch_size = 128
train_generator = train_datagen.flow_from_dataframe(train[:15000], directory=train_dir, x_col='id', y_col='has_cactus', 

                    target_size=(64, 64), color_mode='rgb', classes=None, 

                    class_mode='binary', batch_size=batch_size, 

                    shuffle=True, seed=None, 

                    save_to_dir=None, save_prefix='', save_format='png', 

                    subset=None, interpolation='nearest', drop_duplicates=True)
valid_generator = valid_datagen.flow_from_dataframe(train[15000:], directory=train_dir, x_col='id', y_col='has_cactus', 

                    target_size=(64, 64), color_mode='rgb', classes=None, 

                    class_mode='binary', batch_size=batch_size, 

                    shuffle=True, seed=None, 

                    save_to_dir=None, save_prefix='', save_format='png', 

                    subset=None, interpolation='nearest', drop_duplicates=True)
model = Sequential()
model.add(layers.Conv2D(filters = 32, kernel_size = (3,3), input_shape = (64, 64, 3), activation = 'relu'))

model.add(layers.MaxPooling2D(pool_size = (2,2), padding = 'same'))

model.add(layers.Dropout(0.3))



model.add(layers.Conv2D(filters = 64, kernel_size= (3,3), activation = 'relu'))

model.add(layers.MaxPooling2D(pool_size = (2,2), padding = 'same'))

model.add(layers.Dropout(0.3))





model.add(layers.Conv2D(filters = 128, kernel_size= (3,3), activation = 'relu'))

model.add(layers.MaxPooling2D(pool_size = (2,2), padding = 'same'))

model.add(layers.Dropout(0.3))



model.add(layers.Conv2D(filters = 128, kernel_size= (3,3), activation = 'relu'))

model.add(layers.MaxPooling2D(pool_size = (2,2), padding = 'same'))

model.add(layers.Dropout(0.3))



model.add(layers.Conv2D(filters = 256, kernel_size= (3,3), activation = 'relu'))

model.add(layers.MaxPooling2D(pool_size = (2,2), padding = 'same'))

model.add(layers.Dropout(0.3))



model.add(layers.Flatten())



model.add(layers.Dense(128, activation = 'relu'))

model.add(layers.Dense(1, activation = 'sigmoid'))

model.summary()
optim = Adam(lr=0.0022, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer = optim, loss='binary_crossentropy', metrics=['accuracy'])
filepath = "best_model.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

call_backs_list = [checkpoint]

max_epochs = 60

history = model.fit_generator(

    train_generator,

    steps_per_epoch = 100,

    epochs = max_epochs,

    validation_data = valid_generator,

    callbacks = call_backs_list,

    validation_steps = 50.

)
acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1, len(acc) + 1)



plt.plot(epochs, acc, 'g', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.legend()

plt.figure()



plt.plot(epochs, loss, 'g', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()

plt.show()
model.load_weights("best_model.hdf5")
model.compile(optimizer = optim, loss='binary_crossentropy', metrics=['accuracy'])
un_test_img=[]

count=0

for i in os.listdir("../input/test/test/"):

    un_test_img.append(i)

    count+=1

un_test_image=[]

for i in tqdm(range(count)):

    img = image.load_img('../input/test/test/'+un_test_img[i], target_size=(64,64,3), grayscale=False)

    img = image.img_to_array(img)

    img = img/255

    un_test_image.append(img)

un_test_img_array = np.array(un_test_image)
len(un_test_img)
output = model.predict_classes(un_test_img_array)
submission_save = pd.DataFrame()

submission_save['id'] = un_test_img

submission_save['has_cactus'] = output

submission_save.to_csv('submission.csv', header=True, index=False)
pd.read_csv('submission.csv')