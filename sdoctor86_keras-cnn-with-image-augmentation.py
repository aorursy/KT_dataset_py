import tensorflow as tf

import numpy as np

import pandas as pd

import random

from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Model, Sequential

from tensorflow.keras.layers import Dense,Conv2D,Flatten,Dropout,MaxPooling2D
# load csv

train_df = pd.read_csv('../input/train.csv')
# take a peak

train_df.head()
#disribution is good

train_df.groupby('label').label.count().plot.bar()
# but i found 9s and 4s have trouble, so im going to push more of those into the set

# augmentation will keep from overtraining too much

train_df = train_df.append(train_df[(train_df['label'] == 4) | (train_df['label'] == 9)].sample(1400)).sample(frac=1)

#disribution favors 9's and 4's now

train_df.groupby('label').label.count().plot.bar()
# create label df and remove label from training df

label_df = train_df.label

train_df = train_df.drop('label', axis=1)
DS_SIZE = len(train_df)

BATCH_SIZE = 64



# split in to train and validation sets, for final run train on 99%

x_train, x_val, y_train, y_val = train_test_split(

    np.reshape(train_df.values, (DS_SIZE, 28, 28, 1)), 

    label_df.values, 

    test_size=0.01, 

    random_state=11)



# image augmentation 

train_gen = ImageDataGenerator(

    rescale=1./255,

    shear_range=10,

    zoom_range=0.1,

    width_shift_range=0.05, 

    height_shift_range=0.05,

    rotation_range=15,

    brightness_range=(0.9,1.1),

    fill_mode='constant'

    )



# create flow variable for training, this flows into the fitting method

train_gen_flow = train_gen.flow(

    x_train, 

    y_train, 

    batch_size=BATCH_SIZE

)
# validation will not be augmented, I do this to get a better idea of what to expect on test set

val_gen = ImageDataGenerator(rescale=1./255)



val_gen_flow = val_gen.flow(

    x_val, 

    y_val,

    batch_size=BATCH_SIZE

)
# preview the augmented images

import matplotlib.pyplot as plt

for x_batch, y_batch in train_gen.flow(x_train, y_train, batch_size=9):

    for i in range(0, 9):

        plt.subplot(330 + 1 + i)

        plt.imshow(x_batch[i].reshape(28, 28), cmap=plt.get_cmap('gray'))

    plt.show()

    break
# simple model that puts emphasis on the 2nd pooled filters

def get_model():

    model = Sequential()

    

    model.add(Conv2D(64, (3, 3), input_shape=(28,28,1), activation='relu', padding='same'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(rate=0.4))



    model.add(Conv2D(96, (4, 4), activation='relu'))

    model.add(Conv2D(128, (6, 6), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.5))



    model.add(Flatten())

    model.add(Dense(64))

    model.add(Dropout(0.3))

    model.add(Dense(10, activation="softmax"))

    

    return model



model = get_model()





model.compile(optimizer='adam', 

              loss='sparse_categorical_crossentropy',

              metrics=['acc'])
# fits the model on batches with real-time data augmentation:

history = model.fit_generator(

    train_gen_flow,

    steps_per_epoch=len(x_train)/BATCH_SIZE, 

    epochs=50,

    validation_data = val_gen_flow,

    validation_steps = len(x_val)/BATCH_SIZE

)

# we'll get about 99% after one epoch, but the next 49 epochs gets us as close as we can to perfect

# training accuracy will always be lower than running on test set because of how augmented the data is

# this will keep us being able to generalize well and not overfitting. If you want to only run for 5 epochs, 

# accuracy will be still really good
import matplotlib.pyplot as plt

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.show()
# how this does on non augmented training data

train_df = pd.read_csv('../input/train.csv')



raw_train_image_ds = np.reshape(

    train_df.drop('label', axis=1).values/255.0, 

    (len(train_df), 28, 28, 1)

)



model.evaluate(raw_train_image_ds,train_df.label.values)
# confusion matrix

import seaborn as sns

import matplotlib.pyplot as plt



# from https://gist.github.com/shaypal5/94c53d765083101efc0240d776a23823

def print_confusion_matrix(confusion_matrix, class_names, figsize = (6,6), fontsize=14):

    df_cm = pd.DataFrame(

        confusion_matrix, index=class_names, columns=class_names, 

    )

    fig = plt.figure(figsize=figsize)

    try:

        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")

    except ValueError:

        raise ValueError("Confusion matrix values must be integers.")

    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)

    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    return fig



# predict

preds = np.round(model.predict(raw_train_image_ds))



# confusion

from sklearn.metrics import confusion_matrix

conf_matrix = confusion_matrix(train_df.label.values, np.argmax(preds, axis=1))

conf_matrix_plt = print_confusion_matrix(conf_matrix, ["0","1", "2","3","4","5","6","7","8","9"], figsize = (5,4))



# I'm happy with single digit misses, if there are double digit misses, train for more epochs
# ok time to run on the test set

test_df = pd.read_csv('../input/test.csv')



test_image_ds = np.reshape(

    test_df.values/255.0, 

    (len(test_df), 28, 28, 1)

)



preds = np.round(model.predict(test_image_ds))
# use sample as template

sample_submission_df = pd.read_csv('../input/sample_submission.csv')

sample_submission_df.Label = np.argmax(preds, axis=1)



# verify it looks good

sample_submission_df.head()
sample_submission_df.to_csv('submission.csv', index=False)