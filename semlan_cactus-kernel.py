import keras

from keras.preprocessing import image

from keras.callbacks import ModelCheckpoint



import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



# much prefer ggplot to plt but there seems to be a bug somewhere, can't install?

#!pip install plotnine

#!pip install 'plotnine[all]' 

#from plotnine import *



from IPython.display import Image

import os

#print(os.listdir("../input"))
train_dir="../input/train/train"

test_dir="../input/test/test"

train_data_labels = pd.read_csv('../input/train.csv') # training data and labels

test_data_labels = pd.read_csv('../input/sample_submission.csv') # # test data and labels



print(train_data_labels.shape)

print(test_data_labels.shape)



head_train_data_labels = train_data_labels.head(10)

print(head_train_data_labels)

print(type(head_train_data_labels))

#print(test_data_labels.head(3))
def plot_img_label(df, directory):

    for i, sample in df.iterrows():

        #print(i)

        #print(type(sample))

        #f"{train_dir}/{

        img_file = sample['id']

        img_data = plt.imread(f"{directory}/{img_file}")

        #print(img)

        plt.figure()

        plt.text(10,40, f"has cactus: {sample['has_cactus']}")

        plt.imshow(img_data)

    

plot_img_label(head_train_data_labels, train_dir)

#plt.show()
#ggplot(train_labels, aes('has cactus')) + \

 #   geom_bar(stat = 'count')
model= keras.models.Sequential()

model.add(keras.layers.Conv2D(32,(3,3),activation='relu', input_shape=(32,32,1), padding='same' ))

model.add(keras.layers.MaxPool2D(2,2))

model.add(keras.layers.Conv2D(64,(3,3),activation='relu', padding='same' ))

model.add(keras.layers.MaxPool2D((2,2))) # , padding='same' ))

model.add(keras.layers.Conv2D(128,(3,3),activation='relu', padding='same'))

model.add(keras.layers.MaxPool2D((2,2)))

model.add(keras.layers.Conv2D(256,(3,3),activation='relu', padding='same'))

model.add(keras.layers.MaxPool2D((2,2), padding='same' ))

#model.add(keras.layers.Conv2D(256,(3,3),activation='relu', padding='same'))

#model.add(keras.layers.MaxPool2D((2,2), padding='same' ))

model.add(keras.layers.Dense(512, activation='relu'))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(1,activation='sigmoid'))

print(model.summary())
model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(), metrics= ['acc']) #lr=0.001

#model.compile(loss='binary_crossentropy',optimizer=keras.optimizers.rmsprop(),metrics=['acc'])

# Specify settings for the generators

train_gen = image.ImageDataGenerator( rescale=1./255)# , rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, vertical_flip=True) 

# todo: investigate if rotation & flipping is good? horizontal_flip=True



val_gen = image.ImageDataGenerator(rescale=1./255) # no image augmentation on validation

batch_size = 100



train_data_labels.has_cactus=train_data_labels.has_cactus.astype(str)

# TypeError: If class_mode="binary", y_col="has_cactus" column values must be strings.



validation_samples_num = 2000

train_samples_num = train_data_labels.shape[0] - validation_samples_num



# Create generator functions



# Generator for supplying batches from training data

train_generator = train_gen.flow_from_dataframe(dataframe= train_data_labels.iloc[:train_samples_num],directory=train_dir,x_col='id',

                                            y_col='has_cactus',class_mode='binary',batch_size=batch_size

                                              ,target_size=(32,32), color_mode='grayscale'

                                            )



# Generator for supplying batches from validation data

validation_generator = val_gen.flow_from_dataframe(dataframe= train_data_labels.iloc[train_samples_num:], directory=train_dir,x_col='id',

                                                y_col='has_cactus', class_mode='binary', batch_size=batch_size

                                                ,target_size=(32,32), color_mode='grayscale'

                                                  )



# Generator for supplying batches from test data

test_generator = val_gen.flow_from_dataframe(dataframe= test_data_labels.iloc[:], directory=test_dir,x_col='id',

                                                y_col='has_cactus', class_mode=None, batch_size= batch_size

                                                ,target_size=(32,32), color_mode='grayscale', shuffle=False

                                                  )
#print(train_samples_num / batch_size)

# {epoch:03d}-{acc:03f}-{val_acc:03f}

checkpoint = ModelCheckpoint('best-model.h5', verbose=1, monitor='val_loss',save_best_only=True, mode='auto')

history=model.fit_generator(train_generator, steps_per_epoch = train_samples_num // batch_size ,epochs=15,validation_data=validation_generator, validation_steps = validation_samples_num // batch_size, callbacks=[checkpoint] )


model.load_weights(filepath = 'best-model.h5')

                   

# todo: use flow from directory instead?

test_generator.reset()

predictions = model.predict_generator( test_generator, steps=test_data_labels.shape[0] // batch_size, verbose=1 )

print(predictions[:10])
print(predictions.shape)

print(test_data_labels.shape)

df = pd.DataFrame({'id':test_data_labels['id'], 'has_cactus' : predictions[:,0] })

print(df.shape)

print(df.head())

plot_img_label(df.head(), test_dir)

df.to_csv("submission.csv",index=False)