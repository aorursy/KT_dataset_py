import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

from sklearn.model_selection import train_test_split
from PIL import Image
from imblearn.over_sampling import RandomOverSampler


DIR_NAME = '/kaggle/input/plant-pathology-2020-fgvc7/'

train_set = pd.read_csv(DIR_NAME + 'train.csv')
test_set = pd.read_csv(DIR_NAME + 'test.csv')


target = train_set[['healthy', 'multiple_diseases', 'rust', 'scab']]
test_ids = test_set['image_id']

SIZE = 224

train_len = len(train_set)
test_len = len(test_set)

train_images = np.empty((train_len, SIZE, SIZE, 3))
for i in range(train_len):
    train_images[i] = np.uint8(Image.open(DIR_NAME+ f'images/Train_{i}.jpg').resize((SIZE, SIZE)))
    
test_images = np.empty((test_len, SIZE, SIZE, 3))
for i in range(test_len):
    test_images[i] = np.uint8(Image.open(DIR_NAME + f'images/Test_{i}.jpg').resize((SIZE, SIZE)))
    


x_train, x_test, y_train, y_test = train_test_split(train_images, target.to_numpy(), test_size=0.2, random_state=289)  



ros = RandomOverSampler(random_state=289)

x_train, y_train = ros.fit_resample(x_train.reshape((-1, SIZE * SIZE * 3)), y_train)
x_train = x_train.reshape((-1, SIZE, SIZE, 3))

del train_images


from keras.models import Model, Sequential, load_model, Input
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, LeakyReLU
from keras.regularizers import l2


filters = 32
reg = .0005
SIZE = 224

model = Sequential()

model.add(Conv2D(filters, 3, kernel_regularizer=l2(reg), input_shape=(SIZE, SIZE, 3)))
model.add(LeakyReLU())
model.add(Conv2D(filters, 5, kernel_regularizer=l2(reg)))
model.add(LeakyReLU())
model.add(MaxPooling2D())
model.add(BatchNormalization())
          
model.add(Conv2D(filters, 3, kernel_regularizer=l2(reg)))
model.add(LeakyReLU())
model.add(Conv2D(filters, 5, kernel_regularizer=l2(reg)))
model.add(LeakyReLU())
model.add(MaxPooling2D())
model.add(BatchNormalization())

model.add(Conv2D(filters, 3, kernel_regularizer=l2(reg)))
model.add(LeakyReLU())
model.add(Conv2D(filters, 5, kernel_regularizer=l2(reg)))
model.add(LeakyReLU())
model.add(MaxPooling2D())
model.add(BatchNormalization())

model.add(Conv2D(filters, 3, kernel_regularizer=l2(reg)))
model.add(LeakyReLU())
model.add(Conv2D(filters, 5, kernel_regularizer=l2(reg)))
model.add(LeakyReLU())
model.add(MaxPooling2D())
model.add(BatchNormalization())

model.add(Conv2D(filters, 3, kernel_regularizer=l2(reg)))
model.add(LeakyReLU())
model.add(Conv2D(filters, 5, kernel_regularizer=l2(reg)))
model.add(LeakyReLU())
model.add(MaxPooling2D())
model.add(BatchNormalization())


model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(4, activation='softmax'))

model.summary()
from keras.utils import plot_model

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint



model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['acc']
)

imagegen = ImageDataGenerator(
    rotation_range=40,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True
)


rlr = ReduceLROnPlateau(patience=15, verbose=1)
es = EarlyStopping(patience=50, restore_best_weights=True, verbose=1)
mc = ModelCheckpoint('model.hdf5', save_best_only=True, verbose=0)          
          
history = model.fit_generator(
    imagegen.flow(x_train, y_train, batch_size=32),
    epochs=400,
    steps_per_epoch=x_train.shape[0] // 32,
    verbose=1,
    callbacks=[rlr, es, mc],
    validation_data=(x_test, y_test)
)

from matplotlib import pyplot as plt

# load best model
model = load_model('model.hdf5')
          
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

pred = model.predict(test_images)

res = pd.DataFrame()
res['image_id'] = test_ids
res['healthy'] = pred[:, 0]
res['multiple_diseases'] = pred[:, 1]
res['rust'] = pred[:, 2]
res['scab'] = pred[:, 3]
res.to_csv('submission.csv', index=False)
res.head(50)          