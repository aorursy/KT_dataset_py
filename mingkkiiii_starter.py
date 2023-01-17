# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        break



# Any results you write to the current directory are saved as output.
np.random.seed(1090)
import cv2



root_path = "/kaggle/input/boaz-cs231n-final/training_set/"

        

train_input = []

train_name = []

train_label = []





path = root_path

img_list = os.listdir(path)

for label in img_list:

    print(label)

    for img in os.listdir(path+label+'/'):      

        image = cv2.imread(path+label+'/'+img, cv2.IMREAD_COLOR)

        train_input.append(image)

        train_name.append(img)

        if label == 'dogs':            

            train_label.append(0)

        else:

            train_label.append(1)
X_train = np.array(train_input)

y_train = np.array(train_label)



X_train.shape
print("0번째 이미지의 크기 : ", X_train[0].shape)

print("1번째 이미지의 크기 : ", X_train[1].shape)
from keras.preprocessing.image import ImageDataGenerator





train_data_dir = '/kaggle/input/boaz-cs231n-final/training_set'



batch_size = 64

img_width = 160

img_height = 160



train_datagen = ImageDataGenerator(rescale=1./255,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    validation_split=0.2)



train_generator = train_datagen.flow_from_directory(

    train_data_dir,

    target_size=(img_width, img_height),

    batch_size=batch_size,

    class_mode='categorical',

    subset='training')



validation_generator = train_datagen.flow_from_directory(

    train_data_dir,

    target_size=(img_width, img_height),

    batch_size=batch_size,

    class_mode='categorical',

    subset='validation')
import matplotlib.pyplot as plt

for X_batch, Y_batch in train_generator:

    image = X_batch[0]

    plt.imshow(image)

    print(Y_batch[0])

    break
from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization, GlobalAveragePooling2D





model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(160, 160, 3),padding='same', kernel_initializer='he_normal'))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal'))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal'))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(GlobalAveragePooling2D())

model.add(Dense(32, kernel_initializer='he_normal'))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(Dense(2, activation='softmax')) # 2 because we have cat and dog classes



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

earlystop = EarlyStopping(patience=10)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 

                                            patience=2, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)

callbacks = [earlystop, learning_rate_reduction]



total_train = train_generator.samples

total_validate = validation_generator.samples
epochs= 5

history = model.fit_generator(

    train_generator, 

    epochs=epochs,

    validation_data=validation_generator,

    validation_steps=total_validate//batch_size,

    steps_per_epoch=total_train//batch_size,

    callbacks=callbacks

)
test_filenames = os.listdir("/kaggle/input/boaz-cs231n-final/test_set/")

test_df = pd.DataFrame({

    'filename': test_filenames

})

nb_samples = test_df.shape[0]
IMAGE_SIZE = (160,160)



test_gen = ImageDataGenerator(rescale=1./255)

test_generator = test_gen.flow_from_dataframe(

    test_df, 

    "/kaggle/input/boaz-cs231n-final/test_set/", 

    x_col='filename',

    y_col=None,

    class_mode=None,

    target_size=IMAGE_SIZE,

    batch_size=batch_size,

    shuffle=False

)
predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples/batch_size))
test_df['category'] = np.argmin(predict, axis=-1)
test_df
submission_df = test_df.copy()

submission_df['id'] = submission_df['filename'].str.split('.').str[0]

submission_df['label'] = submission_df['category']

submission_df.drop(['filename', 'category'], axis=1, inplace=True)

submission_df.to_csv('submission.csv', index=False)