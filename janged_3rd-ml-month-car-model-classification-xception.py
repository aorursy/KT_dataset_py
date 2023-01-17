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
import warnings

import seaborn as sns

import matplotlib.pylab as plt

import PIL

from keras import backend as K

from keras.preprocessing.image import ImageDataGenerator

from keras.applications import Xception

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from keras import layers, models, optimizers



warnings.filterwarnings('ignore')



image_size = 299

application = Xception

batch_size = 32
DATA_PATH = '../input'

TRAIN_IMG_PATH = os.path.join(DATA_PATH, 'train')

TEST_IMG_PATH = os.path.join(DATA_PATH, 'test')



df_train = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))

df_test = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))

df_class = pd.read_csv(os.path.join(DATA_PATH, 'class.csv'))
df_train.head()
plt.figure(figsize=(15,6))

sns.countplot('class', data=df_train)
df_train['class'].value_counts()
df_train['class'].value_counts().mean()
df_train['class'].value_counts().describe()
def crop_boxing_img(img_name, margin=16, size=(image_size, image_size)):

    if img_name.split('_')[0] == 'train':

        PATH = TRAIN_IMG_PATH

        data = df_train

    else:

        PATH = TEST_IMG_PATH

        data = df_test



    img = PIL.Image.open(os.path.join(PATH, img_name))

    pos = data.loc[data["img_file"] == img_name, ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']].values.reshape(-1)



    width, height = img.size

    x1 = max(0, pos[0] - margin)

    y1 = max(0, pos[1] - margin)

    x2 = min(pos[2] + margin, width)

    y2 = min(pos[3] + margin, height)



    return img.crop((x1, y1, x2, y2)).resize(size)
nb_train_sample = df_train.shape[0] * 0.7

nb_validation_sample = df_train.shape[0] - nb_train_sample

nb_test_sample = df_test.shape[0]
TRAIN_CROPPED_PATH = '../cropped_train'

VALID_CROPPED_PATH = '../cropped_valid'

TEST_CROPPED_PATH = '../cropped_test'
if (os.path.isdir(TRAIN_CROPPED_PATH) == False):

    os.mkdir(TRAIN_CROPPED_PATH)



if (os.path.isdir(VALID_CROPPED_PATH) == False):

    os.mkdir(VALID_CROPPED_PATH)



if (os.path.isdir(TEST_CROPPED_PATH) == False):

    os.mkdir(TEST_CROPPED_PATH)
df_train['class'] = df_train['class'].astype('str')
for i, row in df_train.iterrows():

    cropped = crop_boxing_img(row['img_file'])

    

    if ( i < nb_train_sample):

        class_path = os.path.join(TRAIN_CROPPED_PATH, df_train['class'][i])

        if(os.path.isdir(class_path) == False):

            os.mkdir(class_path)



        cropped.save(os.path.join(class_path, row['img_file']))

    else:

        class_path = os.path.join(VALID_CROPPED_PATH, df_train['class'][i])

        if(os.path.isdir(class_path) == False):

            os.mkdir(class_path)



        cropped.save(os.path.join(class_path, row['img_file']))



for i, row in df_test.iterrows():

    cropped = crop_boxing_img(row['img_file'])

    cropped.save(os.path.join(TEST_CROPPED_PATH, row['img_file']))
train_datagen = ImageDataGenerator(

    rotation_range=20,

    width_shift_range=0.2,

    height_shift_range=0.2,

    horizontal_flip=True,

    vertical_flip=False,

    zoom_range=0.1,

    fill_mode='nearest'

    )



valid_datagen = ImageDataGenerator()

test_datagen = ImageDataGenerator()
train_generator = train_datagen.flow_from_directory(

    TRAIN_CROPPED_PATH,

    target_size=(image_size, image_size),

    batch_size=batch_size,

    class_mode='categorical',

    seed=2019,

    color_mode='rgb'

)
validation_generator = valid_datagen.flow_from_directory(

    VALID_CROPPED_PATH,

    target_size=(image_size,image_size),

    batch_size=batch_size,

    class_mode='categorical',

    seed=2019,

    color_mode='rgb'

)
test_generator = test_datagen.flow_from_dataframe(

    dataframe=df_test,

    directory=TEST_CROPPED_PATH,

    x_col='img_file',

    y_col=None,

    target_size= (image_size,image_size),

    color_mode='rgb',

    class_mode=None,

    batch_size=batch_size,

    shuffle=False

)
def get_model():

    base_model = application(weights='imagenet', input_shape=(image_size,image_size,3), include_top=False)

    #base_model.trainable = False



    model = models.Sequential()

    model.add(base_model)

    model.add(layers.GlobalAveragePooling2D())

    model.add(layers.Dense(1024, activation='relu'))

    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(196, activation='softmax'))

    model.summary()



    #optimizer = optimizers.SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)

    optimizer = optimizers.RMSprop(lr=0.0001)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])



    return model
model = get_model()
model_path = '../model/'

if not os.path.exists(model_path):

    os.mkdir(model_path)

    

model_path = model_path + 'best_model.hdf5'
patient = 2

callbacks1 = [

    EarlyStopping(monitor='val_loss', patience=patient, mode='min', verbose=1),

    ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = patient / 2, min_lr=0.00001, verbose=1, mode='min'),

    ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min'),

    ]
def get_steps(num_samples, batch_size):

    if (num_samples % batch_size) > 0:

        return (num_samples // batch_size) + 1

    else:

        return num_samples // batch_size
history = model.fit_generator(

    train_generator,

    steps_per_epoch=get_steps(nb_train_sample, batch_size),

    epochs=100,

    validation_data=validation_generator,

    validation_steps=get_steps(nb_validation_sample, batch_size),

    verbose=1,

    callbacks = callbacks1

)
acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1, len(acc) + 1)



plt.plot(epochs, acc, 'bo', label='Training Acc')

plt.plot(epochs, val_acc, 'b', label='Validation Acc')

plt.title('Training and validation accuracy')

plt.legend()

plt.show()



plt.plot(epochs, loss, 'bo', label='Traing loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Trainging and validation loss')

plt.legend()

plt.show()
model.load_weights(model_path)

test_generator.reset()



prediction = model.predict_generator(

    generator=test_generator,

    steps = get_steps(nb_test_sample, batch_size),

    verbose=1

)


predicted_class_indices=np.argmax(prediction, axis=1)



# Generator class dictionary mapping

labels = (train_generator.class_indices)

labels = dict((v,k) for k,v in labels.items())

predictions = [labels[k] for k in predicted_class_indices]



submission = pd.read_csv(os.path.join(DATA_PATH, 'sample_submission.csv'))

submission["class"] = predictions

submission.to_csv("submission.csv", index=False)

submission.head()