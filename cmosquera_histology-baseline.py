from google.colab import drive

drive.mount('/content/gdrive',force_remount=True)

import pandas as pd



folder = '/content/gdrive/My Drive/kaggle-histology/'

df=pd.read_csv(folder+"train_labels.csv")

df['id']=[str(x)+'.tif' for x in df['id']]

df['label']=[str(x) for x in df['label']]
example_imgfile = df['id'][0]



print(example_imgfile)
from skimage.io import imread

from matplotlib import pyplot as plt



img = imread(folder + 'train/' + example_imgfile)

plt.imshow(img)

print(img.shape)

import os

print('Tamaño conjunto train',len(os.listdir(folder + "train/")))

print('Tamaño conjunto test',len(os.listdir(folder + "test/")))


from keras_preprocessing.image import ImageDataGenerator

datagen=ImageDataGenerator(rescale=1./255,validation_split=0.25)

train_generator=datagen.flow_from_dataframe(dataframe=df, directory=folder + "train/", 

                                            x_col="id", y_col="label", class_mode="binary", 

                                            subset='training',

                                            target_size=(50,50), batch_size=8)

valid_generator=datagen.flow_from_dataframe(dataframe=df, directory=folder + "train/", 

                                            x_col="id", y_col="label", class_mode="binary", 

                                            subset='validation',

                                            target_size=(50,50), batch_size=8)



from keras.models import Sequential

from keras.layers import Conv2D,Activation,MaxPooling2D,Dropout,Flatten,Dense

model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',

                 input_shape=(50,50,3)))

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

model.add(Dense(1, activation='softmax'))





from keras import optimizers

model.compile(optimizer=optimizers.rmsprop(lr=0.0001),loss="binary_crossentropy", metrics=["accuracy"])

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size

STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size



hist = model.fit_generator(generator=train_generator,

                    steps_per_epoch=STEP_SIZE_TRAIN,

                    validation_data=valid_generator,

                    validation_steps=STEP_SIZE_VALID,

                    epochs=1)
test_datagen=ImageDataGenerator(rescale=1./255)

test_generator=datagen.flow_from_directory(directory=folder + "test/", 

                                            target_size=(50,50), batch_size=1)



STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

preds = [int(x[0]) for x in model.predict_generator(generator=test_generator,steps=STEP_SIZE_TEST)]
submission = pd.DataFrame()

submission['id'] = os.listdir(folder + "test/allclasses")

submission['label'] = preds

submission.head()
submission.to_csv(folder + "submission.csv",index=False)