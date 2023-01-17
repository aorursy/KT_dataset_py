import cv2

from IPython.display import Image

from IPython.display import Image

from keras.preprocessing import image

from keras import optimizers

from keras import layers,models

from keras.applications.imagenet_utils import preprocess_input

import matplotlib.pyplot as plt

import seaborn as sns

from keras import regularizers

from keras.preprocessing.image import ImageDataGenerator

from tqdm import tqdm_notebook
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_dir="../input/train/train"

test_dir=r"../input/test/test"

train=pd.read_csv('../input/train.csv')

df_test=pd.read_csv('../input/sample_submission.csv')
train.head(5)
train.has_cactus=train.has_cactus.astype(str)
train.shape[0],train.shape[1]
train['has_cactus'].value_counts()
Image(os.path.join(train_dir,train.iloc[1,0]),width=250,height=250)
datagen=ImageDataGenerator(rescale=1./255)

batch_size=150
train_generator=datagen.flow_from_dataframe(dataframe=train[:15001],directory=train_dir,x_col='id',

                                            y_col='has_cactus',class_mode='binary',batch_size=batch_size,

                                            target_size=(150,150))





validation_generator=datagen.flow_from_dataframe(dataframe=train[15000:],directory=train_dir,x_col='id',

                                                y_col='has_cactus',class_mode='binary',batch_size=50,

                                                target_size=(150,150))
model=models.Sequential()

model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))

model.add(layers.MaxPool2D((2,2)))

model.add(layers.Conv2D(64,(3,3),activation='relu',input_shape=(150,150,3)))

model.add(layers.MaxPool2D((2,2)))

model.add(layers.Conv2D(128,(3,3),activation='relu',input_shape=(150,150,3)))

model.add(layers.MaxPool2D((2,2)))

model.add(layers.Conv2D(128,(3,3),activation='relu',input_shape=(150,150,3)))

model.add(layers.MaxPool2D((2,2)))

model.add(layers.Flatten())

model.add(layers.Dense(512,activation='relu'))

# model.add(layers.Dense(64,activation='relu'))

model.add(layers.Dense(1,activation='sigmoid'))
model.summary()
model.compile(loss='binary_crossentropy',optimizer=optimizers.rmsprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0),

metrics=['acc'])

# model.compile(loss='binary_crossentropy',optimizer=optimizers.adam(),metrics=['acc'])

# model.compile(loss='binary_crossentropy',optimizer=optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

#               ,metrics=['acc'])

epochs=10

history=model.fit_generator(train_generator,steps_per_epoch=100,epochs=epochs,

                            validation_data=validation_generator,validation_steps=50)
acc=history.history['acc']  ##getting  accuracy of each epochs

epochs_=range(0,epochs)    

plt.plot(epochs_,acc,label='training accuracy')

plt.xlabel('no of epochs')

plt.ylabel('accuracy')



acc_val=history.history['val_acc']  ##getting validation accuracy of each epochs

plt.scatter(epochs_,acc_val,label="validation accuracy")

plt.title("no of epochs vs accuracy")

plt.legend()
acc=history.history['loss']    ##getting  loss of each epochs

epochs_=range(0,epochs)

plt.plot(epochs_,acc,label='training loss')

plt.xlabel('No of epochs')

plt.ylabel('loss')



acc_val=history.history['val_loss']  ## getting validation loss of each epochs

plt.scatter(epochs_,acc_val,label="validation loss")

plt.title('no of epochs vs loss')

plt.legend()
test_generetor = ImageDataGenerator(rescale = 1./255)
test_gen = test_generetor.flow_from_directory(directory=r"../input/test",target_size=(150,150),

                                    color_mode="rgb",batch_size=1,class_mode=None,shuffle=False,seed=42)
test_features = []

Test_images = []

for img_id in tqdm_notebook(os.listdir(test_dir)):

    test_features.append(cv2.resize(cv2.imread(os.path.join(test_dir,img_id)),(150,150)))     

    Test_images.append(img_id)

test_features = np.asarray(test_features)

test_features = test_features.astype('float32')

test_features /= 255
test_features.shape
test_predictions = model.predict(test_features)

submissions = pd.DataFrame(test_predictions, columns=['has_cactus'])

submissions['has_cactus'] = submissions['has_cactus'].apply(lambda x: 1 if x > 0.75 else 0)

submissions['id'] = ''

cols = submissions.columns.tolist()

cols = cols[-1:] + cols[:-1]

submissions=submissions[cols]



# STEP_SIZE_TEST=test_gen.n//test_gen.batch_size

# test_gen.reset()

# pred=model.predict_generator(test_gen,

# steps=STEP_SIZE_TEST,

# verbose=1)
submissions.head()
for i, img in enumerate(Test_images):

    submissions.set_value(i,'id',img)
submissions.head()
submissions.to_csv('submission.csv',index=False)