import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os



PATH = '/kaggle/input/plant-pathology-2020-fgvc7/'



train = pd.read_csv(PATH + 'train.csv')

test = pd.read_csv(PATH + 'test.csv')



target = train[['healthy', 'multiple_diseases', 'rust', 'scab']]

test_ids = test['image_id']



train_len = train.shape[0]

test_len = test.shape[0]



train.describe()
from PIL import Image

from tqdm.notebook import tqdm



SIZE = 224



train_images = np.empty((train_len, SIZE, SIZE, 3))

for i in tqdm(range(train_len)):

    train_images[i] = np.uint8(Image.open(PATH + f'images/Train_{i}.jpg').resize((SIZE, SIZE)))

    

test_images = np.empty((test_len, SIZE, SIZE, 3))

for i in tqdm(range(test_len)):

    test_images[i] = np.uint8(Image.open(PATH + f'images/Test_{i}.jpg').resize((SIZE, SIZE)))



train_images.shape, test_images.shape
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(train_images, target.to_numpy(), test_size=0.2, random_state=289) 



x_train.shape, x_test.shape, y_train.shape, y_test.shape
from imblearn.over_sampling import RandomOverSampler



ros = RandomOverSampler(random_state=289)



x_train, y_train = ros.fit_resample(x_train.reshape((-1, SIZE * SIZE * 3)), y_train)

x_train = x_train.reshape((-1, SIZE, SIZE, 3))

x_train.shape, y_train.sum(axis=0)
import gc



del train_images

gc.collect()
from keras.models import Model, Sequential, load_model, Input

from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, LeakyReLU

from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

from keras.utils import plot_model

from keras.regularizers import l2



rlr = ReduceLROnPlateau(patience=15, verbose=1)

es = EarlyStopping(patience=35, restore_best_weights=True, verbose=1)

mc = ModelCheckpoint('model.hdf5', save_best_only=True, verbose=0)



filters = 32

reg = .0005



model = Sequential()



for i in range(5):

    model.add(Conv2D(filters, 3, kernel_regularizer=l2(reg), input_shape=(SIZE, SIZE, 3)))

    model.add(LeakyReLU())

    

    model.add(Conv2D(filters, 3, kernel_regularizer=l2(reg)))

    model.add(LeakyReLU())

    

    if i != 4:

        model.add(Conv2D(filters, 5, kernel_regularizer=l2(reg)))

        model.add(LeakyReLU())

        

    model.add(MaxPooling2D())

    model.add(Dropout(0.5))

    model.add(BatchNormalization())



    filters *= 2



model.add(Flatten())

model.add(Dense(16, activation='relu'))

model.add(Dropout(0.5))

model.add(BatchNormalization())

model.add(Dense(4, activation='softmax'))



model.summary()
from keras.preprocessing.image import ImageDataGenerator



model.compile(

    optimizer='adam',

    loss='categorical_crossentropy',

    metrics=['acc']

)



imagegen = ImageDataGenerator(

    rotation_range=20,

    zoom_range=0.2,

    width_shift_range=0.2,

    height_shift_range=0.2,

    horizontal_flip=True,

    vertical_flip=True

)



history = model.fit_generator(

    imagegen.flow(x_train, y_train, batch_size=32),

    epochs=400,

    steps_per_epoch=x_train.shape[0] // 32,

    verbose=0,

    callbacks=[rlr, es, mc],

    validation_data=(x_test, y_test)

)

# load best model

model = load_model('model.hdf5')
from matplotlib import pyplot as plt



h = history.history



offset = 5

epochs = range(offset, len(h['loss']))



plt.figure(1, figsize=(20, 6))



plt.subplot(121)

plt.xlabel('epochs')

plt.ylabel('loss')

plt.plot(epochs, h['loss'][offset:], label='train')

plt.plot(epochs, h['val_loss'][offset:], label='val')

plt.legend()



plt.subplot(122)

plt.xlabel('epochs')

plt.ylabel('accuracy')

plt.plot(h[f'acc'], label='train')

plt.plot(h[f'val_acc'], label='val')

plt.legend()



plt.show()
from sklearn.metrics import roc_auc_score



pred_test = model.predict(x_test)

roc_sum = 0

for i in range(4):

    score = roc_auc_score(y_test[:, i], pred_test[:, i])

    roc_sum += score

    print(f'{score:.3f}')



roc_sum /= 4

print(f'totally:{roc_sum:.3f}')
pred = model.predict(test_images)



res = pd.DataFrame()

res['image_id'] = test_ids

res['healthy'] = pred[:, 0]

res['multiple_diseases'] = pred[:, 1]

res['rust'] = pred[:, 2]

res['scab'] = pred[:, 3]

res.to_csv('submission.csv', index=False)

res.head(40)