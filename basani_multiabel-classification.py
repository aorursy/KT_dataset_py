# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

'''for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

'''

# Any results you write to the current directory are saved as output.
import keras



from keras import regularizers, optimizers

from keras.optimizers import SGD,adam

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten,Activation

from keras.layers import Conv2D, MaxPooling2D

from keras.utils import to_categorical

from keras.preprocessing import image

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from tqdm import tqdm

%matplotlib inline
train = pd.read_csv('../input/preprocessed-image-data/train_preprocessed.csv')    # reading the csv file

train = train.drop('Unnamed: 0',axis=1)

train['imageId']= train['imageId'].astype(str)+'.jpg'

validation = pd.read_csv('../input/preprocessed-image-data/validation_preprocessed.csv')    # reading the csv file

validation = validation.drop('Unnamed: 0',axis=1)

validation['imageId']=validation['imageId'].apply(str)+'.jpg'

train.head()
columns = train.columns.drop(['imageId','labelId'])

print(columns)
train.head()
data = []

for file in sorted(os.listdir('../input/image-data/data/Data/test/')):

    #print(file)

    data.append(file)

result= pd.DataFrame(data,columns = ['imageId'])

result.head()
from tensorflow.keras.preprocessing.image import ImageDataGenerator



train_dataGen = ImageDataGenerator(rescale = 1./255,

                                  shear_range = 0.2,

                                  zoom_range = 0.2,

                                  horizontal_flip = True)

test_dataGen = ImageDataGenerator(rescale = 1./255)





train_generator = train_dataGen.flow_from_dataframe(

                                        dataframe = train,

                                        directory="../input/image-data/data/Data/train/",x_col='imageId',

                                        y_col=columns, 

                                        class_mode="other",

                                        target_size=(256,256),

                                        seed=42,

                                        batch_size=32)

valid_generator=test_dataGen.flow_from_dataframe(

                                        dataframe=validation[:2500],

                                        directory="../input/image-data/data/Data/validation/", 

                                        x_col="imageId",

                                        y_col=columns,

                                        batch_size=32,

                                        seed=42,

                                        shuffle=True,

                                        class_mode="other",

                                        target_size=(256,256))
test_generator = test_dataGen.flow_from_dataframe( directory = "../input/image-data/data/Data/validation/",

                                                  dataframe = validation[2500:],

                                                  x_col = "imageId",

                                                  target_size=(256,256),

                                                  seed=42,

                                                  batch_size=5,

                                                  class_mode=None)
train_image = []

for i in tqdm(range(1,10)):

    try:

        img = image.load_img('../input/image-data/data/Data/train/'+str(train['imageId'][i]),target_size=(256,256,3))

        img = image.img_to_array(img)

        print(img.shape)

        img = img/255

        train_image.append(img)

    except (FileNotFoundError):

        pass

X = np.array(train_image)
print(plt.imshow(X[5]))
model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',

                 input_shape=(256,256,3)))

model.add(Activation('relu'))

model.add(Conv2D(32, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))

model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(512))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(228, activation='sigmoid'))

model.compile(optimizer='adam',loss="binary_crossentropy",metrics=["accuracy"])
model = Sequential()

model.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu", input_shape=(256,256,3)))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu"))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(700, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(228, activation='sigmoid'))

sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=False)



model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size

STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size

STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

history = model.fit_generator(generator=train_generator,

                    steps_per_epoch=STEP_SIZE_TRAIN,

                    validation_data=valid_generator,

                    validation_steps=STEP_SIZE_VALID,

                    epochs=3

)
# list all data in history

print(history.history.keys())

# summarize history for accuracy

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
test_generator.reset()

pred=model.predict_generator(test_generator,

steps=STEP_SIZE_TEST,

verbose=1)
pred.shape
pred_bool = (pred >0.3)
predictions = pred_bool.astype(int)

predictions.shape

#columns should be the same order of y_col

results=pd.DataFrame(predictions, columns=columns)



results.head()



results["imageId"]=test_generator.filenames

ordered_cols=columns.insert(0,'imageId')



results=results[ordered_cols]#To get the same column order

results.to_csv("results.csv",index=False)
#image = '3001.jpg'



bt = results[columns].apply(lambda x: x > 0)

results['labelId'] = bt.apply(lambda x: list(columns[x.values]), axis=1)
results['labelId'][1]