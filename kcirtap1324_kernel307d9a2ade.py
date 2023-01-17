# All used modules

import numpy as np

import pandas as pd

from keras.models import Sequential

from keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D

from keras.preprocessing.image import ImageDataGenerator

from keras import applications, optimizers

import matplotlib.pyplot as plt
# Change to 1 for verbose

DEBUG = 1
# Batch size

batch_size = 250



epochs = 20

epoch_steps = 63

valid_steps = 14
# Dataset path

train_dir = '../input/train/train'

test_dir = '../input/train/train'

testing_dir = "../input/train"
# Verbose progress

if DEBUG == 1:

    print('Reading datasets...')

    

# Get all training cactus image file names

train_cactus = pd.read_csv('../input/train.csv')

test_cactus = pd.read_csv('../input/sample_submission.csv')

train_cactus['has_cactus'] = np.where(train_cactus['has_cactus'] == 1, 'yes', 'no')



# Validate dataframe

if DEBUG == 1:

    print(train_cactus.head())
train_datagen=ImageDataGenerator(

    rescale=1./255, 

    validation_split=0.10,

    rotation_range=40,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    fill_mode='nearest')



train_generator=train_datagen.flow_from_dataframe(

    dataframe=train_cactus,

    directory=train_dir,

    x_col='id',

    y_col='has_cactus',

    class_mode='binary',

    batch_size=batch_size,

    shuffle=True,

    subset='training',

    target_size=(32,32))



valid_generator = train_datagen.flow_from_dataframe(

    dataframe = train_cactus,

    directory = train_dir,

    x_col="id",

    y_col="has_cactus",

    target_size=(32,32),

    subset="validation",

    batch_size=125,

    shuffle=True,

    class_mode="binary"

)
test_datagen = ImageDataGenerator(

    rescale=1/255

)



test_generator = test_datagen.flow_from_directory(

    testing_dir,

    target_size=(32,32),

    batch_size=1,

    shuffle=False,

    class_mode=None

)
densenet = applications.densenet.DenseNet201(

    include_top = False,

    weights ='imagenet',

    input_shape = (32,32,3))

model = Sequential()

model.add(densenet)

model.add(GlobalAveragePooling2D())

model.add(Dense(units = 120, activation='relu'))

model.add(Dense(units = 84, activation='relu'))

model.add(Dense(units = 1, activation='sigmoid'))
model.compile(optimizer=optimizers.Adam(

    lr=0.0001), loss = 'binary_crossentropy', metrics=['accuracy'])
history = model.fit_generator(

    train_generator,

    epochs = epochs,

    steps_per_epoch = epoch_steps,

    validation_data = valid_generator,

    validation_steps = valid_steps)
acc, loss = history.history['acc'], history.history['loss']

val_acc, val_loss = history.history['val_acc'], history.history['val_loss']



epochs = len(acc)



if DEBUG == 1:

    plt.plot(range(epochs), acc, color='red', label='Training Accuracy')

    plt.plot(range(epochs), val_acc, color='green', label='Validation Accuracy')

    plt.legend()

    plt.title('Accuracy over Training')

    plt.xlabel('Epochs')

    plt.ylabel('Accuracy')

    plt.show()



    plt.plot(range(epochs), loss, color='red', label='Training Loss')

    plt.plot(range(epochs), val_loss, color='green', label='Validation Loss')

    plt.legend()

    plt.title('Loss over Training')

    plt.xlabel('Epochs')

    plt.ylabel('Loss')

    plt.show()
import statistics



print(statistics.mean(history.history['acc']))

print(statistics.stdev(history.history['acc']))
preds = model.predict_generator(

    test_generator,

    steps=len(test_generator.filenames)

)
image_ids = [name.split('/')[-1] for name in test_generator.filenames]

predictions = preds.flatten()

data = {'id': image_ids, 'has_cactus':predictions} 

submission = pd.DataFrame(data)

if DEBUG == 1:

    print(submission.head())
submission.to_csv("submission.csv", index=False)

from IPython.display import HTML

import base64



def create_download_link( df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv()

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = f'<a target="_blank">{title}</a>'

    return HTML(html)



create_download_link(submission)