import numpy as np

import pandas as pd 

import os

import keras

import matplotlib.pyplot as plt

import seaborn as sns



print(keras.__version__)



print("{} paintings".format(len(os.listdir("../input/resized/resized"))))
df = pd.read_csv('../input/artists.csv')

df.head()
from skimage.transform import resize

from keras.preprocessing import image



img_dir = '../input/images/images/'

files = os.listdir(img_dir)

img_size = (300,300)

input_shape = [*img_size, 3]



def load_img(path):

    img = image.load_img(path=path, target_size=img_size)

    return np.asarray(img, dtype="int32" )/255
datagen = keras.preprocessing.image.ImageDataGenerator(

    featurewise_center=False,

    featurewise_std_normalization=False,

    rescale=1.0/255,

    rotation_range=0,

    width_shift_range=0.1,

    height_shift_range=0.1,

    horizontal_flip=True,

    validation_split=0.1)



train_generator = datagen.flow_from_directory(

        img_dir,

        target_size=img_size,

        batch_size=12,

        class_mode='categorical',

        subset="training")

validation_generator = datagen.flow_from_directory(

        img_dir,

        target_size=img_size,

        batch_size=12,

        class_mode='categorical',

        subset="validation")



num_classes = len(train_generator.class_indices)
plt.hist(train_generator.classes)

plt.hist(validation_generator.classes)
from keras.applications.xception import Xception

from keras.models import Model

from keras.layers import GlobalAveragePooling2D, Dense



base_model = Xception(include_top=False, weights='imagenet')

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(1024, activation='relu', name="dense_1024")(x)

predictions = Dense(num_classes, activation='softmax', name="predictions")(x)



model = Model(inputs=base_model.input, outputs=predictions)
for i, layer in enumerate(base_model.layers):

    print(i, layer.name)
# freeze some layers, only train the last blocks

for layer in base_model.layers[:31]:

    layer.trainable = True

for layer in base_model.layers[86:]:

    layer.trainable = True
%%time

from keras.callbacks import *

from keras.optimizers import *



callbacks = [

    EarlyStopping(monitor='val_loss', patience=6),

    ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True),

    ReduceLROnPlateau(monitor='val_loss', factor=0.2, min_lr=0.001)

]



model.compile(optimizer=SGD(lr=0.01),

              loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(

    train_generator, 

    callbacks=callbacks, 

    validation_data=validation_generator, 

    epochs=16,

    steps_per_epoch=500,

    validation_steps=80,

    workers=2)
model.load_weights('best_model.h5')



model.evaluate_generator(datagen.flow_from_directory(

        img_dir,

        target_size=img_size,

        batch_size=16,

        class_mode='categorical'), steps=100)
from keras.applications.resnet50 import preprocess_input, decode_predictions

from keras.models import Model



class ImgVectorizer:



    def __init__(self, model, dense_layer_name):

        self.intermediate_layer_model = Model(

            inputs=model.input, 

            outputs=model.get_layer(dense_layer_name).output

        )



    def to_vector(self, imgs):

        """ Gets a vector embedding from an image

        :param image_path: path to image on filesystem

        :returns: numpy ndarray

        """

        batch = np.array(imgs)

        intermediate_output = self.intermediate_layer_model.predict(batch)

        return intermediate_output

    

vectorizer = ImgVectorizer(model=model, dense_layer_name="dense_1024")
def batched(batch_size, iterable):

    batch = []

    for item in iterable:

        batch.append(item)

        if len(batch) == batch_size:

            yield batch

            batch = []

    if len(batch) > 0:

        yield batch
import glob

import re



p = re.compile(".*/images/(?P<name>.*)/(?P<img>.*)\.jpg")



def img_dict_generator():

    for file in glob.glob(img_dir + '*/*'):

        match = p.match(file)

        if match:

            artist = p.match(file).group("name")

            yield {'artist': artist, 'file': file, 'vector': []}

        

df = pd.DataFrame(img_dict_generator())
%%time



def add_vector_field(dataframe):

    dataframe['file'].values



for indexes in batched(16, df.index):

    df_batch = df.loc[indexes]

    imgs = [load_img(file) for file in df_batch['file'].values]

    vectors = vectorizer.to_vector(imgs)

    df_batch['vector'] = [tuple(list) for list in vectors]

    df.update(df_batch)
df.to_hdf('df_with_vectors.h5', key="vectors")

df.to_csv('df_with_vectors.csv')