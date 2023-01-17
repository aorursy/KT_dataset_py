import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sn

import random



from keras.preprocessing.image import ImageDataGenerator, load_img

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split



%matplotlib inline

import os

os.listdir('../input')
os.listdir('../input/data/data')
train_data=pd.read_csv('../input/data/data/train.csv',dtype=str)

train_data.head()
train_data.dtypes
train_data.count()
test_data=pd.read_csv('../input/data/data/test_ApKoW4T.csv')

test_data.head()
test_data.count()
sample_sub=pd.read_csv('../input/data/data/sample_submission_ns2btKE.csv')

sample_sub.head()
sample_sub.count()
sample_sub.tail()
train_data.tail()
train_data['category'].value_counts()
train_data['category'].value_counts().plot.bar()

plt.show()
filenames = os.listdir("../input/data/data/images")

sample = random.choice(filenames)

image = load_img("../input/data/data/images/"+sample)

plt.imshow(image)

plt.show()
FAST_RUN = False

IMAGE_WIDTH=128

IMAGE_HEIGHT=128

IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)

IMAGE_CHANNELS=3 # RGB color
from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization



model = Sequential()



model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.summary()
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
earlystop = EarlyStopping(patience=10)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=2, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
callbacks = [earlystop, learning_rate_reduction]
train_df, validate_df = train_test_split(train_data, test_size=0.20, random_state=42)

train_df = train_df.reset_index(drop=True)

validate_df = validate_df.reset_index(drop=True)

test_df = test_data.reset_index(drop=True)
train_df['category'].value_counts()
train_df['category'].value_counts().plot.bar()

plt.show()
validate_df['category'].value_counts()
validate_df['category'].value_counts().plot.bar()

plt.show()
total_train = train_df.shape[0]

total_validate = validate_df.shape[0]

batch_size=15
train_datagen = ImageDataGenerator(

    rotation_range=15,

    rescale=1./255,

    shear_range=0.1,

    zoom_range=0.2,

    horizontal_flip=True,

    width_shift_range=0.1,

    height_shift_range=0.1

)



train_generator = train_datagen.flow_from_dataframe(

    dataframe=train_df, 

    directory="../input/data/data/images/", 

    x_col='image',

    y_col="category",

    target_size=IMAGE_SIZE,

    class_mode='categorical',

    batch_size=batch_size

)
validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_dataframe(

    validate_df, 

    "../input/data/data/images/", 

    x_col='image',

    y_col='category',

    target_size=IMAGE_SIZE,

    class_mode='categorical',

    batch_size=batch_size

)
example_df = train_df.sample(n=1).reset_index(drop=True)

example_generator = train_datagen.flow_from_dataframe(

    example_df, 

    "../input/data/data/images/", 

    x_col='image',

    y_col='category',

    target_size=IMAGE_SIZE,

    class_mode='categorical'

)
plt.figure(figsize=(12, 12))

for i in range(0, 15):

    plt.subplot(5, 3, i+1)

    for X_batch, Y_batch in example_generator:

        image = X_batch[0]

        plt.imshow(image)

        break

plt.tight_layout()

plt.show()
epochs=3 if FAST_RUN else 50

history = model.fit_generator(

    train_generator, 

    epochs=epochs,

    validation_data=validation_generator,

    validation_steps=total_validate//batch_size,

    steps_per_epoch=total_train//batch_size,

    callbacks=callbacks

)
model.save_weights("model.h5")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

ax1.plot(history.history['loss'], color='b', label="Training loss")

ax1.plot(history.history['val_loss'], color='r', label="validation loss")

ax1.set_xticks(np.arange(1, epochs, 1))

ax1.set_yticks(np.arange(0, 1, 0.1))



ax2.plot(history.history['acc'], color='b', label="Training accuracy")

ax2.plot(history.history['val_acc'], color='r',label="Validation accuracy")

ax2.set_xticks(np.arange(1, epochs, 1))



legend = plt.legend(loc='best', shadow=True)

plt.tight_layout()

plt.show()
test_gen = ImageDataGenerator(rescale=1./255)

test_generator = test_gen.flow_from_dataframe(

    test_df, 

    "../input/data/data/images/", 

    x_col='image',

    y_col=None,

    class_mode=None,

    target_size=IMAGE_SIZE,

    batch_size=batch_size,

    shuffle=False

)
nb_samples = test_df.shape[0]

nb_samples
test_generator.reset()

predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples/batch_size))
predicted_class_indices=np.argmax(predict,axis=1)

predicted_class_indices
labels = (train_generator.class_indices)

labels = dict((v,k) for k,v in labels.items())

predictions = [labels[k] for k in predicted_class_indices]

filenames=test_df.image

results=pd.DataFrame({"image":filenames,

                      "category":predictions})

results.to_csv("results2.csv",index=False)