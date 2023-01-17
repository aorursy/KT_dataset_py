import pandas as pd

import numpy as np



from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, Activation



from keras.preprocessing.image import ImageDataGenerator 

#from keras.applications import VGG19



import warnings

warnings.filterwarnings("ignore")
train = pd.read_csv("../input/train.csv")

train.head()
train["has_cactus"] = train["has_cactus"].map(lambda x:str(x))

train.shape
%%time

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1, 

                                  horizontal_flip=True, vertical_flip=True)



train_generator = train_datagen.flow_from_dataframe(dataframe=train,

                                                   directory = "../input/train/train",

                                                   x_col="id", y_col="has_cactus",

                                                   batch_size=32, shuffle=True,

                                                   class_mode="binary",

                                                   target_size=(32, 32),

                                                   subset="training")
%%time

val_generator = train_datagen.flow_from_dataframe(dataframe=train,

                                                 directory = "../input/train/train",

                                                 x_col="id", y_col="has_cactus",

                                                 batch_size=32, shuffle=True,

                                                 class_mode="binary",

                                                 target_size=(32, 32),

                                                 subset="validation")
from keras import applications
base_model = applications.VGG16(weights='imagenet', 

                     include_top=False, 

                     input_shape=(32, 32, 3))
model = Sequential()

model.add(base_model)

model.add(Flatten())

model.add(Dense(256, use_bias=True))

model.add(BatchNormalization())

model.add(Activation("relu"))

model.add(Dropout(0.5))

model.add(Dense(256,activation='relu'))

model.add(BatchNormalization())

model.add(Dense(16, activation='tanh'))

model.add(Dense(1, activation='sigmoid'))

model.summary()
model.compile(loss='binary_crossentropy',

             optimizer='adam',

             metrics=['accuracy'])
Model = model.fit_generator(generator=train_generator,

                           validation_data=val_generator,

                           validation_steps=int(train.shape[0]/32),

                           steps_per_epoch=int(train.shape[0]/32),

                           epochs=20, verbose=2)
test_dir="../input/test/test/"
import os

import cv2

from tqdm import tqdm, tqdm_notebook



X_test = []

X_image = []



for image in tqdm_notebook(os.listdir(test_dir)):

    X_test.append(cv2.imread(test_dir+image))

    X_image.append(image)

X_test = np.array(X_test)

X_test = X_test/255.0
testPredict = model.predict(X_test)
submission=pd.DataFrame(testPredict,columns=['has_cactus'])
submission['id'] = ''

cols=list(submission.columns)

cols = cols[-1:] + cols[:-1]

submission=submission[cols]

for i, img in enumerate(X_image):

    submission.set_value(i,'id',img)
submission.to_csv('submission.csv',index=False)