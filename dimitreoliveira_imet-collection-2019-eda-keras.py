import os

import cv2

import warnings

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from keras import optimizers

from keras.models import Sequential

from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Activation, BatchNormalization



%matplotlib inline

sns.set(style="whitegrid")

warnings.filterwarnings("ignore")



# Set seeds to make the experiment more reproducible.

from tensorflow import set_random_seed

from numpy.random import seed

set_random_seed(0)

seed(0)
train = pd.read_csv('../input/train.csv')

labels = pd.read_csv('../input/labels.csv')

test = pd.read_csv('../input/sample_submission.csv')



print('Number of train samples: ', train.shape[0])

print('Number of test samples: ', test.shape[0])

print('Number of labels: ', labels.shape[0])

display(train.head())

display(labels.head())
attribute_ids = train['attribute_ids'].values

attributes = []

for item_attributes in [x.split(' ') for x in attribute_ids]:

    for attribute in item_attributes:

        attributes.append(int(attribute))

        

att_pd = pd.DataFrame(attributes, columns=['attribute_id'])

att_pd = att_pd.merge(labels)

top30 = att_pd['attribute_name'].value_counts()[:30].to_frame()

N_unique_att = att_pd['attribute_id'].nunique()

print('Number of unique attributes: ', N_unique_att)

f, ax = plt.subplots(figsize=(12, 8))

ax = sns.barplot(y=top30.index, x="attribute_name", data=top30, palette="rocket", order=reversed(top30.index))

ax.set_ylabel("Surface type")

ax.set_xlabel("Count")

sns.despine()

plt.show()
att_pd['tag'] = att_pd['attribute_name'].apply(lambda x:x.split('::')[0])

gp_att = att_pd.groupby('tag').count()



print('Number of attributes groups: ', gp_att.shape[0])

f, ax = plt.subplots(figsize=(12, 8))

ax = sns.barplot(y=gp_att.index, x="attribute_name", data=gp_att, palette="rocket")

ax.set_ylabel("Attribute group")

ax.set_xlabel("Count")

sns.despine()

plt.show()
train['Number of Tags'] = train['attribute_ids'].apply(lambda x: len(x.split(' ')))

f, ax = plt.subplots(figsize=(12, 8))

ax = sns.countplot(x="Number of Tags", data=train, palette="GnBu_d")

ax.set_ylabel("Surface type")

sns.despine()

plt.show()
sns.set_style("white")

count = 1

plt.figure(figsize=[20,20])

for img_name in os.listdir("../input/train/")[:20]:

    img = cv2.imread("../input/train/%s" % img_name)[...,[2, 1, 0]]

    plt.subplot(5, 5, count)

    plt.imshow(img)

    plt.title("Item %s" % count)

    count += 1

    

plt.show()
train["id"] = train["id"].apply(lambda x:x+".png")

test["id"] = test["id"].apply(lambda x:x+".png")

train["attribute_ids"] = train["attribute_ids"].apply(lambda x:x.split(" "))
# Model parameters

BATCH_SIZE = 128

EPOCHS = 30

LEARNING_RATE = 0.0001

HEIGHT = 64

WIDTH = 64

CANAL = 3

N_CLASSES = N_unique_att

classes = list(map(str, range(N_CLASSES)))
model = Sequential()



model.add(Conv2D(filters=32, kernel_size=(5,5),padding='Same', input_shape=(HEIGHT, WIDTH, CANAL)))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Conv2D(filters=32, kernel_size=(5,5),padding='Same'))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.5))



model.add(Conv2D(filters=64, kernel_size=(4,4),padding='Same'))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Conv2D(filters=64, kernel_size=(4,4),padding='Same'))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.5))



model.add(Flatten())



model.add(Dense(1024))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Dropout(0.5))



model.add(Dense(N_CLASSES, activation="sigmoid"))

model.summary()



optimizer = optimizers.adam(lr=LEARNING_RATE)

model.compile(optimizer=optimizer , loss="binary_crossentropy", metrics=["accuracy"])
train_datagen=ImageDataGenerator(rescale=1./255, validation_split=0.25)

test_datagen = ImageDataGenerator(rescale=1./255)



train_generator=train_datagen.flow_from_dataframe(

    dataframe=train,

    directory="../input/train",

    x_col="id",

    y_col="attribute_ids",

    batch_size=BATCH_SIZE,

    shuffle=True,

    class_mode="categorical",

    classes=classes,

    target_size=(HEIGHT, WIDTH),

    subset='training')



valid_generator=train_datagen.flow_from_dataframe(

    dataframe=train,

    directory="../input/train",

    x_col="id",

    y_col="attribute_ids",

    batch_size=BATCH_SIZE,

    shuffle=True,

    class_mode="categorical",    

    classes=classes,

    target_size=(HEIGHT, WIDTH),

    subset='validation')



test_generator = test_datagen.flow_from_dataframe(  

        dataframe=test,

        directory = "../input/test",    

        x_col="id",

        target_size = (HEIGHT, WIDTH),

        batch_size = 1,

        shuffle = False,

        class_mode = None)
STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size

STEP_SIZE_VAL = valid_generator.n // valid_generator.batch_size



history = model.fit_generator(generator=train_generator,

                    steps_per_epoch=STEP_SIZE_TRAIN,

                    validation_data=valid_generator,

                    validation_steps=STEP_SIZE_VAL,

                    epochs=EPOCHS,

                    verbose=2)
sns.set_style("whitegrid")

fig, (ax1, ax2) = plt.subplots(1, 2, sharex='col', figsize=(20,7))



ax1.plot(history.history['acc'], label='Train Accuracy')

ax1.plot(history.history['val_acc'], label='Validation accuracy')

ax1.legend(loc='best')

ax1.set_title('Accuracy')



ax2.plot(history.history['loss'], label='Train loss')

ax2.plot(history.history['val_loss'], label='Validation loss')

ax2.legend(loc='best')

ax2.set_title('Loss')



plt.xlabel('Epochs')

sns.despine()

plt.show()
test_generator.reset()

n_steps = len(test_generator.filenames)

preds = model.predict_generator(test_generator, steps = n_steps)
predictions = []

for pred_ar in preds:

    valid = ''

    for idx, pred in enumerate(pred_ar):

        if pred > 0.3:  # Using 0.3 as threshold

            if len(valid) == 0:

                valid += str(idx)

            else:

                valid += (' %s' % idx)

    if len(valid) == 0:

        valid = str(np.argmax(pred_ar))

    predictions.append(valid)
filenames=test_generator.filenames

results=pd.DataFrame({'id':filenames, 'attribute_ids':predictions})

results['id'] = results['id'].map(lambda x: str(x)[:-4])

results.to_csv('submission.csv',index=False)

results.head(10)