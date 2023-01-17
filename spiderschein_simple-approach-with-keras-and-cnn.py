import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Conv2D, Dropout, MaxPooling2D, Dense, Flatten, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split

batch_size = 64
epochs = 100
# Input data files are available in the "../input/" directory.

train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
train_x_data = (train_data.iloc[:,1:].values).astype('float32')
train_y_data = (train_data.iloc[:,0].values).astype('float32')
test_x_data = (test_data.values).astype('float32')

print('Split 10% of the data for evaluatio')
train_x_data, eval_x_data, train_y_data, eval_y_data = train_test_split(train_x_data, train_y_data, test_size=0.1)

print('Normalization')
train_x_data, eval_x_data, test_x_data = map(lambda data: data / 255, [train_x_data, eval_x_data, test_x_data])

print('Reshapig X_data')
train_x_data, eval_x_data, test_x_data = map(lambda data: data.reshape(data.shape[0], 28, 28, 1), [train_x_data, eval_x_data, test_x_data])

print('Reshaping Y_data')
train_y_data, eval_y_data = map(to_categorical, [train_y_data, eval_y_data])

del train_data, test_data


#for i in range(330, 340):
#    plt.subplot(330 + (i+1))
#    plt.imshow(train_x_data[i], cmap=plt.get_cmap('gray'))
#
#    plt.title(train_y_data[i])
dataGenerator = ImageDataGenerator(
    featurewise_center=False,  
    samplewise_center=False, 
    featurewise_std_normalization=False, 
    samplewise_std_normalization=False,
    zca_whitening=False, 
    rotation_range=10, 
    zoom_range = 0.1, 
    shear_range = 0.1,
    width_shift_range=0.1,  
    height_shift_range=0.1,
    horizontal_flip=False,
    vertical_flip=False
)

dataGenerator.fit(train_x_data)

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(filters=64, kernel_size=(2, 2), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters=64, kernel_size=(2, 2), padding='same', activation='relu'))
model.add(BatchNormalization())
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

model.summary() 
history = model.fit_generator(
    dataGenerator.flow(train_x_data, train_y_data, batch_size=batch_size),
    validation_data=(eval_x_data, eval_y_data),
    epochs=epochs,
    steps_per_epoch=len(train_x_data)/batch_size
)
plt.plot(history.history['acc'], label='Acc')
plt.plot(history.history['val_acc'], label='Val_Acc')
plt.legend()

results = model.predict(test_x_data)

results = np.argmax(results, axis=1)
results = pd.Series(results, name='Label')

submission = pd.concat([pd.Series(range(1, 28001), name='ImageId'), results], axis=1)
submission.to_csv('output.csv', index=False)
