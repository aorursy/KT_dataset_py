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



# Any results you write to the current directory are saved as output.

train_df = pd.read_csv('/kaggle/input/train.csv')
#convert to str to use class mode binary( categories )

train_df['has_cactus'] = train_df['has_cactus'].apply(lambda x: str(x))
#split train-validation 

train_split = 0.4 

validation_df = train_df.sample(n=int(train_split*len(train_df)))
len(validation_df)
len(train_df)
#trim validation set data points

train_df = train_df[~train_df['id'].isin(validation_df['id'])]
#check how much sample size we have for each category

print(validation_df['has_cactus'].value_counts())

print(train_df['has_cactus'].value_counts())
from keras import models

from keras import layers



# this architecture was taken from the deep-learning with keras book.

# conv2d+MaxPool layer -> flatten -> dense layer(relu) -> dense_layer(sigmoid)

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))

model.add(layers.MaxPooling2D((2, 2)))

#another conv layer can be added if needed

#model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())

model.add(layers.Dense(16, activation='relu'))

model.add(layers.Dense(1,activation='sigmoid'))

model.build()

model.summary()

from keras.preprocessing.image import ImageDataGenerator



# All images will be rescaled by 1./255

train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(dataframe=train_df,directory='/kaggle/input/train/train/',x_col='id',y_col='has_cactus',target_size=(32,32),class_mode='binary',batch_size=20)





# All images will be rescaled by 1./255

validator_datagen = ImageDataGenerator(rescale=1./255)

validator_generator = validator_datagen.flow_from_dataframe(dataframe=validation_df,directory='/kaggle/input/train/train/',x_col='id',y_col='has_cactus',target_size=(32,32),class_mode='binary',batch_size=20)

len(set(train_df['id']) & set(validation_df['id']))
#binary_crossentropy works well comparing probability distributions

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit_generator(train_generator,steps_per_epoch=len(train_df)//20,epochs=20,validation_data=validator_generator,validation_steps=len(validation_df)//20)

import matplotlib.pyplot as plt



def plot_history(history):

    acc = history.history['acc']

    val_acc = history.history['val_acc']

    loss = history.history['loss']

    val_loss = history.history['val_loss']



    epochs = range(len(acc))



    plt.plot(epochs, acc, 'bo', label='Training acc')

    plt.plot(epochs, val_acc, 'b', label='Validation acc')

    plt.title('Training and validation accuracy')

    plt.legend()



    plt.figure()



    plt.plot(epochs, loss, 'bo', label='Training loss')

    plt.plot(epochs, val_loss, 'b', label='Validation loss')

    plt.title('Training and validation loss')

    plt.legend()



    plt.show()





    

plot_history(history)
# run with test set

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory('/kaggle/input/test', target_size=(32, 32),class_mode=None, batch_size=1,shuffle=False)



Y_pred = model.predict_generator(test_generator,steps=len(filenames))

filenames = list(map(lambda x: x.replace('test/',''),test_generator.filenames))

test_df = pd.DataFrame()

test_df['id']=filenames

test_df['has_cactus']=Y_pred

test_df[['id','has_cactus']].to_csv('submission_baseline.csv',index=False)