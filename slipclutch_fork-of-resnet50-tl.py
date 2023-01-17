# load dependencies
import numpy as np 
import pandas as pd
import os
import shutil
import cv2
import matplotlib.pyplot as plt
# load keras modules
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam
from keras.regularizers import l2

os.getcwd()
#os.chdir('/Users/Aron/Kaggle/plant_pathology')
local_dir = '/Users/Aron/Kaggle/plant_pathology/plant-pathology-2020-fgvc7'
kaggle_dir = '/kaggle/input/plant-pathology-2020-fgvc7/'

sample_submission = pd.read_csv('../input/plant-pathology-2020-fgvc7/sample_submission.csv')
test = pd.read_csv(kaggle_dir + 'test.csv')
train = pd.read_csv(kaggle_dir + 'train.csv')

csv = pd.read_csv('../input/results/submission_densenet.csv')
csv.head()

img_size = 256
#Resize the training images
from tqdm.notebook import tqdm
train_image = []
for name in tqdm(train['image_id']):
    path='../input/plant-pathology-2020-fgvc7/images/'+name+'.jpg'
    img=cv2.imread(path)
    image=cv2.resize(img,(img_size,img_size),interpolation=cv2.INTER_LINEAR)
    train_image.append(image)
#resize the testing images
from tqdm.notebook import tqdm
test_image = []
for name in tqdm(test['image_id']):
    path='../input/plant-pathology-2020-fgvc7/images/'+name+'.jpg'
    img=cv2.imread(path)
    image=cv2.resize(img,(img_size,img_size),interpolation=cv2.INTER_LINEAR)
    test_image.append(image)
len(test_image)

#reshape the data
#shape the training images to work for keras.
x_train = np.asarray(train_image, dtype=np.float32)
x_train = x_train/255

x_test = np.asarray(test_image, dtype=np.float32)
x_test = x_test/255

y = train.iloc[:,1:5]
# turn the labels into an arrray
y_train = np.array(y.values, dtype='float32')



x_test.shape, x_test.shape
x_train, x_val, y_train, y_val = train_test_split(x_train, 
                                                  y_train, 
                                                  test_size = 0.20, 
                                                  random_state = 403 )
x_train.shape, x_val.shape, y_train.shape, y_val.shape
from imblearn.over_sampling import RandomOverSampler

oversample = RandomOverSampler(random_state=403)

x_train, y_train = oversample.fit_resample(x_train.reshape((-1, img_size * img_size * 3)), y_train)
x_train = x_train.reshape((-1, img_size, img_size, 3))
x_train.shape, y_train.sum(axis=0)
batch_size = 32
# create the model. but add in the "with"
#with tpu_strategy.scope():
model = Sequential()

model.add(Conv2D(32, (3, 3), kernel_regularizer=l2(.001), input_shape=(img_size, img_size,3)))
model.add(Activation('relu'))

model.add(Conv2D(32, (3, 3), kernel_regularizer=l2(.001)))
model.add(Activation('relu'))

model.add(Conv2D(32, (5,5), kernel_regularizer=l2(.001)))
model.add(Activation('relu'))

model.add(MaxPooling2D())
model.add(Dropout(0.3))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), kernel_regularizer=l2(.001), input_shape=(img_size, img_size,3)))
model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3), kernel_regularizer=l2(.001)))
model.add(Activation('relu'))

model.add(Conv2D(64, (5,5), kernel_regularizer=l2(.001)))
model.add(Activation('relu'))

model.add(MaxPooling2D())
model.add(Dropout(0.3))
model.add(BatchNormalization())

model.add(Conv2D(128, (3, 3), kernel_regularizer=l2(.001), input_shape=(img_size, img_size,3)))
model.add(Activation('relu'))

model.add(Conv2D(128, (3, 3), kernel_regularizer=l2(.001)))
model.add(Activation('relu'))

model.add(Conv2D(128, (5,5), kernel_regularizer=l2(.001)))
model.add(Activation('relu'))

model.add(MaxPooling2D())
model.add(Dropout(0.5))
model.add(BatchNormalization())

model.add(Conv2D(256, (3, 3), kernel_regularizer=l2(.001), input_shape=(img_size, img_size,3)))
model.add(Activation('relu'))

model.add(Conv2D(256, (3, 3), kernel_regularizer=l2(.001)))
model.add(Activation('relu'))

model.add(Conv2D(256, (5,5), kernel_regularizer=l2(.001)))
model.add(Activation('relu'))


model.add(MaxPooling2D())
model.add(Dropout(0.5))
model.add(BatchNormalization())

model.add(Flatten())  
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(4, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer= Adam(lr=.001),
              metrics=['accuracy']) 
model.summary()
# now create the data generator
datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
#fit the model
history = model.fit_generator(
        datagen.flow(x_train, y_train, batch_size=32),
        steps_per_epoch=x_train.shape[0] // batch_size,
        epochs=70,
        validation_data=(x_val, y_val))
        
model.save_weights('cnnDataGenerator_weights.h5')

predict= model.predict(x_test)
prediction = np.ndarray(shape = (test.shape[0],4), dtype = np.float32)
for row in range(test.shape[0]):
    for col in range(4):
        if predict[row][col] == max(predict[row]):
            prediction[row][col] = 1
        else:
            prediction[row][col] = 0
prediction = pd.DataFrame(prediction)
prediction.columns = ['healthy', 'multiple_diseases', 'rust', 'scab']
df = pd.concat([test.image_id, prediction], axis = 1)
df.to_csv('submission.csv', index = False)
from IPython.display import FileLink
FileLink(r'submission.csv')
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training', 'validataion'], loc='upper left')
plt.show()
