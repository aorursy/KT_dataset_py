import sys, os, shutil, glob



import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns



# from tqdm import tqdm

from itertools import chain



# Printing models in ipynb

from PIL import Image

from scipy import ndimage

from keras.utils.vis_utils import model_to_dot, plot_model

from IPython.display import SVG



from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.preprocessing import LabelEncoder

from sklearn.pipeline import Pipeline



from skimage.io import imread, imshow, imread_collection, concatenate_images

from skimage.transform import resize



import tensorflow as tf



import keras

from keras import regularizers, optimizers

from keras.applications import VGG16, VGG19

from keras.models import Model, load_model, Sequential

from keras.layers import Dense, Activation, Flatten, Dropout

from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization

from keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing.image import array_to_img, img_to_array

from keras.preprocessing.image import load_img

from keras import backend as K

from keras.callbacks import ModelCheckpoint

from keras.wrappers.scikit_learn import KerasClassifier

from keras.utils import np_utils



import warnings

warnings.filterwarnings('ignore')

# OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.

# os.environ['KMP_DUPLICATE_LIB_OK']='True'



np.random.seed(42)
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))
%ls ../input/dataset/dataset_updated
%ls ../input/dataset/dataset_updated/training_set/drawings/ | head -5
%ls ../input/musemart/dataset_updated/training_set/drawings/ | head -5
imshow('../input/musemart/dataset_updated/training_set/drawings/1677_mainfoto_05.jpg');
imshow('../input/dataset/dataset_updated/training_set/drawings/1677_mainfoto_05.jpg');
src_dirs_0 = ['dataset', 'musemart']

src_dirs_1 = ['training_set', 'validation_set']

src_dirs_2 = ['sculpture', 'iconography', 'engraving', 'drawings', 'painting']
# copying files from musemart to dataset (merge data)

for sub_dir in src_dirs_1:

    for d in src_dirs_2:

        src_dir = src_dirs_0[1] + '/' + sub_dir + '/' + d

        files = os.listdir(src_dir)

        

        dst_dir = src_dirs_0[0] + '/' + sub_dir + '/' + d

        

        for file in files:

            shutil.copy(os.path.join(src_dir, file), os.path.join(dst_dir, file))
img_width, img_height = 150, 150



categories = ['drawings', 'engraving', 'iconography' ,'painting' ,'sculpture']



train_path = 'dataset/train/'

valid_path = 'dataset/validation/'

test_path  = 'dataset/test/'
def show_images_for_art(art_type="drawings", num_pics=10):

    assert art_type in categories

    

    pic_dir = os.path.join(train_path, art_type)

    pic_files = [os.path.join(pic_dir, filename) for filename in os.listdir(pic_dir)]



    ncols = 5

    nrows = (num_pics - 1) // ncols + 1

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 4))

    

    fig.set_size_inches((20, nrows * 5))

    ax = ax.ravel()

    

    for pic, ax in zip(pic_files[:num_pics], ax):

        img = imread(pic)

        ax.imshow(img, resample=True)

    

    plt.show();

    

show_images_for_art(art_type="drawings")
# Just have a look at the categories itself, one image shall be ok



fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(16, 4))



for i, cat in enumerate(categories):

    cat_path = os.path.join(train_path, cat)

    img_name = os.listdir(cat_path)[0]

    

    img = imread(os.path.join(cat_path, img_name))

    img = resize(img, (img_width, img_height, 3), mode='reflect')

    

    ax[i].imshow(img, resample=True)

    ax[i].set_title(cat)

    

plt.show();
n_imgs = []

for cat in categories:

    files = os.listdir(os.path.join(train_path, cat))

    n_imgs += [len(files)]



plt.figure(figsize=(16, 8))

plt.bar([_ for _ in range(5)], n_imgs, tick_label=categories)

plt.show();
num_train_sample = 0

for i, cat in enumerate(categories):

    cat_path = os.path.join(train_path, cat)

    num_train_sample += len(os.listdir(cat_path))

    

print('Total number of training samples: {}'.format(num_train_sample))
num_test_sample = 0

for i, cat in enumerate(categories):

    cat_path = os.path.join(test_path, cat)

    num_test_sample += len(os.listdir(cat_path))

    

print('Total number of test samples: {}'.format(num_test_sample))
num_validation_sample = 0

for i, cat in enumerate(categories):

    cat_path = os.path.join(valid_path, cat)

    num_validation_sample += len(os.listdir(cat_path))

    

print('Total number of validation samples: {}'.format(num_validation_sample))
nb_train_samples = 2000

nb_validation_samples = 800

epochs = 50

batch_size = 16
# this is the augmentation configuration we will use for training

train_datagen = ImageDataGenerator(

        rescale=1./255,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True)



# this is the augmentation configuration we will use for testing:

# only rescaling

datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(

        train_path,

        target_size=(img_width, img_height),

        batch_size=batch_size,

        class_mode='categorical')



valid_generator = datagen.flow_from_directory(

        valid_path,

        target_size=(img_width, img_height),

        batch_size=batch_size,

        class_mode='categorical')



test_generator = datagen.flow_from_directory(

        test_path,

        target_size=(img_width, img_height),

        batch_size=batch_size,

        class_mode='categorical')
if K.image_data_format() == 'channels_first':

    input_shape = (3, img_width, img_height)

else:

    input_shape = (img_width, img_height, 3)
model = Sequential([

    Conv2D(32, (3, 3), input_shape=input_shape, activation='relu'),

    MaxPooling2D(pool_size=(2, 2)),

    Dropout(0.2),

    

    Conv2D(32, (3, 3), activation='relu'),

    MaxPooling2D(pool_size=(2, 2)),

    Dropout(0.2),

    

    Conv2D(64, (3, 3), activation='relu'),

    MaxPooling2D(pool_size=(2, 2)),

    Dropout(0.2),

    

    Flatten(),

    Dense(128, activation='relu'),

    Dropout(0.2),

    Dense(64, activation='relu'),

    Dropout(0.2),

    Dense(5, activation='softmax')

])



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
SVG(model_to_dot(model, show_layer_names=False, show_shapes=True).create(prog='dot', format='svg'))
%%time



train_result = model.fit_generator(

            train_generator,

            steps_per_epoch=num_train_sample // batch_size,

            epochs=epochs,

            validation_data=valid_generator,

            validation_steps=num_validation_sample // batch_size,

            use_multiprocessing=True)
model.save('CNN_base_run1.h5')
sns.set(style="white", palette="muted", color_codes=True)



fig, ax = plt.subplots(1, 2, figsize=(16, 6), sharex=True)



ax[0].plot(train_result.history['loss'], label="Loss")

ax[0].plot(train_result.history['val_loss'], label="Validation loss")

ax[0].set_title('Loss')

ax[0].set_xlabel('Epoch')

ax[0].set_ylabel('Loss')

ax[0].legend()



ax[1].plot(train_result.history['acc'], label="Accuracy")

ax[1].plot(train_result.history['val_acc'], label="Validation accuracy")

ax[1].set_title('Accuracy')

ax[1].set_xlabel('Epoch')

ax[1].set_ylabel('Accuracy')

ax[1].legend()

plt.tight_layout()



plt.show();
test_loss, test_acc = model.evaluate_generator(test_generator, steps=32)

y_hat_test = model.predict_generator(test_generator, steps=32)



print('Generated {} predictions'.format(len(y_hat_test)))

print('Test accuracy: {:.2f}%'.format(test_acc * 100))
# Load the VGG19 network

vgg_model = VGG19(include_top=False, weights='imagenet', input_shape=input_shape)

vgg_model.summary()
model = Sequential([

    vgg_model,

    Flatten(),

    Dense(32, activation='relu'),

    Dense(64, activation='relu'),

    Dense(128, activation='relu'),

    Dense(64, activation='relu'),

    Dense(5, activation='softmax')

])



vgg_model.trainable = False



# Check what layers are trainable

for layer in model.layers:

    print(layer.name, layer.trainable)

    

# model.summary()
SVG(model_to_dot(model, show_layer_names=False, show_shapes=True).create(prog='dot', format='svg'))
%%time



# Compilation

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



# Fitting the Model

train_result = model.fit_generator(

            train_generator,

            steps_per_epoch=num_train_sample // batch_size,

            epochs=epochs,

            validation_data=valid_generator,

            validation_steps=num_validation_sample // batch_size,

            use_multiprocessing=True)
model.save('VGG19_Feature_Engineered.h5')
sns.set(style="white", palette="muted", color_codes=True)



fig, ax = plt.subplots(1, 2, figsize=(16, 6), sharex=True)



ax[0].plot(train_result.history['loss'], label="Loss")

ax[0].plot(train_result.history['val_loss'], label="Validation loss")

ax[0].set_title('Loss')

ax[0].set_xlabel('Epoch')

ax[0].set_ylabel('Loss')

ax[0].legend()



ax[1].plot(train_result.history['acc'], label="Accuracy")

ax[1].plot(train_result.history['val_acc'], label="Validation accuracy")

ax[1].set_title('Accuracy')

ax[1].set_xlabel('Epoch')

ax[1].set_ylabel('Accuracy')

ax[1].legend()

plt.tight_layout()



plt.show();
test_loss, test_acc = model.evaluate_generator(test_generator, steps=32, use_multiprocessing=True)

y_hat_test = model.predict_generator(test_generator, steps=32, use_multiprocessing=True)



print('Generated {} predictions'.format(len(y_hat_test)))

print('Test accuracy: {:.2f}%'.format(test_acc * 100))
# Load the VGG19 network

vgg_model = VGG19(include_top=False, weights='imagenet', input_shape=input_shape)

vgg_model.summary()
%%time

datagen = ImageDataGenerator(rescale=1. / 255)



generator = datagen.flow_from_directory(

    train_path,

    target_size=(img_width, img_height),

    batch_size=batch_size,

    class_mode='categorical',

    shuffle=False)



bottleneck_features_train = vgg_model.predict_generator(

    generator, 500, use_multiprocessing=True)



# Save the output as a numpy array

np.save(open('bottleneck_features_train.npy', 'wb'),

        bottleneck_features_train)
%%time

generator = datagen.flow_from_directory(

    valid_path,

    target_size=(img_width, img_height),

    batch_size=batch_size,

    class_mode='categorical',

    shuffle=False)



bottleneck_features_validation = vgg_model.predict_generator(

    generator, 60, use_multiprocessing=True, verbose=1)



# Save the output as a numpy array

np.save(open('bottleneck_features_validation.npy', 'wb'), 

        bottleneck_features_validation)
bottleneck_features_train.shape, bottleneck_features_validation.shape
train_data = np.load(open('bottleneck_features_train.npy', 'rb'))

# train_labels = np.array([0] * 4000 + [1] * 4000)

a = np.zeros(shape=(8000, 3))

b = np.ones(shape=(8000, 2))

train_labels = np.concatenate((a, b), axis=1)



validation_data = np.load(open('bottleneck_features_validation.npy', 'rb'))

# validation_labels = np.array([0] * 480 + [1] * 480)

a = np.zeros(shape=(960, 3))

b = np.ones(shape=(960, 2))

validation_labels = np.concatenate((a, b), axis=1)
train_data.shape, train_labels.shape
model = Sequential([

    Flatten(input_shape=train_data.shape[1:]),

    Dense(256, activation='relu'),

    Dropout(0.2),

    Dense(5, activation='softmax')

])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
%%time

train_result = model.fit(train_data, train_labels,

                         epochs=epochs,

                         batch_size=batch_size,

                         validation_data=(validation_data, validation_labels)

                        )
model.save_weights('bottleneck_fc_model.h5')
sns.set(style="white", palette="muted", color_codes=True)



fig, ax = plt.subplots(1, 2, figsize=(16, 6), sharex=True)



ax[0].plot(train_result.history['loss'], label="Loss")

ax[0].plot(train_result.history['val_loss'], label="Validation loss")

ax[0].set_title('Loss')

ax[0].set_xlabel('Epoch')

ax[0].set_ylabel('Loss')

ax[0].legend()



ax[1].plot(train_result.history['acc'], label="Accuracy")

ax[1].plot(train_result.history['val_acc'], label="Validation accuracy")

ax[1].set_title('Accuracy')

ax[1].set_xlabel('Epoch')

ax[1].set_ylabel('Accuracy')

ax[1].legend()

plt.tight_layout()



plt.show();
img_width, img_height = 128, 128

input_shape = (img_height, img_width, 3)



categories = ['drawings', 'engraving', 'iconography' ,'painting' ,'sculpture']



train_path = 'dataset/train/'

valid_path = 'dataset/validation/'

test_path = 'dataset/test/'
category_embeddings = {

    'drawings': 0,

    'engraving': 1,

    'iconography': 2,

    'painting': 3,

    'sculpture': 4

}
train_data = [(file, cat) for cat in categories for file in glob.glob(train_path + cat + '/*')]

test_data = [(file, cat) for cat in categories for file in glob.glob(train_path + cat + '/*')]
train_data[:5]
def load_dataset(tuples_list):

    indexes = np.arange(len(tuples_list))

    np.random.shuffle(indexes)

    

    X = []

    y = []

    

    cpt = 0

    for i in range(len(indexes)):

        t = tuples_list[indexes[i]]

        try:

            # skimage

            img = imread(t[0])

            img = resize(img, input_shape, mode='reflect')

            X += [img]

            

            y_tmp = [0 for _ in range(len(categories))]

            y_tmp[category_embeddings[t[1]]] = 1

            y += [y_tmp]

        except OSError:

            pass

        

        cpt += 1

        if cpt % 1000 == 0:

            print("Processed {} images".format(cpt))



    return np.array(X), np.array(y)
X_train, y_train = load_dataset(train_data)

X_valid, y_valid = load_dataset(test_data)
train_datagen = ImageDataGenerator(rotation_range=20, zoom_range=0.15, horizontal_flip=True)

train_datagen.fit(X_train)
# model = Sequential([

#     Conv2D(32, (3, 3), padding='same', input_shape=input_shape, activation='relu'),

#     Conv2D(32, (3, 3), activation='relu'),

#     MaxPooling2D(pool_size=(2, 2)),

    

#     Dropout(0.25),

#     Conv2D(64, (3, 3), padding='same', activation='relu'),

#     Conv2D(64, (3, 3), activation='relu'),

#     MaxPooling2D(pool_size=(2, 2)),

    

#     Dropout(0.25),

#     Flatten(),

#     Dense(256, activation='relu'),

#     Dropout(0.5),

#     Dense(5, activation='sigmoid')

# ])

model = Sequential([

    Conv2D(32, (3, 3), input_shape=input_shape, activation='relu'),

    MaxPooling2D(pool_size=(2, 2)),

    Dropout(0.2),

    

    Conv2D(32, (3, 3), activation='relu'),

    MaxPooling2D(pool_size=(2, 2)),

    Dropout(0.2),

    

    Conv2D(64, (3, 3), activation='relu'),

    MaxPooling2D(pool_size=(2, 2)),

    Dropout(0.2),

    

    Flatten(),

    Dense(128, activation='relu'),

    Dropout(0.2),

    Dense(64, activation='relu'),

    Dropout(0.2),

    Dense(5, activation='softmax')

])



model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["accuracy"])
train_generator = train_datagen.flow(X_train, y_train, batch_size=32)

train_result = model.fit_generator(generator=train_generator, validation_data=(X_valid, y_valid),

                                  epochs=50, steps_per_epoch=len(X_train)/32, verbose=1, 

                                  use_multiprocessing=True)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))



ax[0].plot(train_result.history['loss'], label="Loss")

ax[0].plot(train_result.history['val_loss'], label="Validation loss")

ax[0].set_title('Loss')

ax[0].set_xlabel('Epoch')

ax[0].set_ylabel('Loss')

ax[0].legend()



ax[1].plot(train_result.history['acc'], label="Accuracy")

ax[1].plot(train_result.history['val_acc'], label="Validation accuracy")

ax[1].set_title('Accuracy')

ax[1].set_xlabel('Epoch')

ax[1].set_ylabel('Accuracy')

ax[1].legend()



plt.tight_layout()

plt.show();
# Let's look at more metrics

from sklearn.metrics import classification_report



X_test = []

y_test = []

for t in test_data:

    try:

        img = skimage.io.imread(os.path.join(t[0]))

        img = skimage.transform.resize(img, input_shape, mode='reflect')

        X_test += [img]

        y_test += [category_embeddings[t[1]]]

    except OSError:

        pass



X_test = np.array(X_test)

y_test = np.array(y_test)



pred = model.predict(X_test, verbose=1)



y_pred = np.argmax(pred, axis=1)

print(classification_report(y_test, y_pred))
from sklearn.metrics import confusion_matrix



c_matrix = confusion_matrix(y_test, y_pred)

plt.imshow(c_matrix, cmap=plt.cm.Blues)

plt.title("Confusion matrix")

plt.colorbar()

plt.show();



print(c_matrix)
categories = os.listdir("dataset/train/")

categories
dataset = pd.DataFrame(columns=[categories])

dataset.insert(loc=0, column='filename', value=None)

dataset.head()
df1 = dataset.copy()

df2 = dataset.copy()

df3 = dataset.copy()

df4 = dataset.copy()

df5 = dataset.copy()

df5.head()
myList = [file.split("/", 1)[1] for file in glob.glob('dataset/train/sculpture/*')]

dataset = pd.DataFrame(data=myList, columns=['filename'])

dataset[categories] = pd.DataFrame([[1, 0, 0, 0, 0]], index=dataset.index)



myList = [file.split("/", 1)[1] for file in glob.glob('dataset/train/iconography/*')]

df2 = pd.DataFrame(data=myList, columns=['filename'])

df2[categories] = pd.DataFrame([[0, 1, 0, 0, 0]], index=df2.index)

dataset = dataset.append(df2)



myList = [file.split("/", 1)[1] for file in glob.glob('dataset/train/engraving/*')]

df3 = pd.DataFrame(data=myList, columns=['filename'])

df3[categories] = pd.DataFrame([[0, 0, 1, 0, 0]], index=df3.index)

dataset = dataset.append(df3)



myList = [file.split("/", 1)[1] for file in glob.glob('dataset/train/drawings/*')]

df4 = pd.DataFrame(data=myList, columns=['filename'])

df4[categories] = pd.DataFrame([[0, 0, 0, 1, 0]], index=df4.index)

dataset = dataset.append(df3)



myList = [file.split("/", 1)[1] for file in glob.glob('dataset/train/painting/*')]

df5 = pd.DataFrame(data=myList, columns=['filename'])

df5[categories] = pd.DataFrame([[0, 0, 0, 0, 1]], index=df5.index)

dataset = dataset.append(df5)
dataset.shape
dataset.reset_index(drop=True, inplace=True)
dataset.tail()
valid_dataset = pd.DataFrame(columns=[categories])

valid_dataset.insert(loc=0, column='filename', value=None)

valid_dataset.head()
del df1, df2, df3, df4, df5
df1 = valid_dataset.copy()

df2 = valid_dataset.copy()

df3 = valid_dataset.copy()

df4 = valid_dataset.copy()

df5 = valid_dataset.copy()

df5.head()
myList = [file.split("/", 1)[1] for file in glob.glob('dataset/validation/sculpture/*')]

valid_dataset = pd.DataFrame(data=myList, columns=['filename'])

valid_dataset[categories] = pd.DataFrame([[1, 0, 0, 0, 0]], index=dataset.index)



myList = [file.split("/", 1)[1] for file in glob.glob('dataset/validation/iconography/*')]

df2 = pd.DataFrame(data=myList, columns=['filename'])

df2[categories] = pd.DataFrame([[0, 1, 0, 0, 0]], index=df2.index)

valid_dataset = valid_dataset.append(df2)



myList = [file.split("/", 1)[1] for file in glob.glob('dataset/validation/engraving/*')]

df3 = pd.DataFrame(data=myList, columns=['filename'])

df3[categories] = pd.DataFrame([[0, 0, 1, 0, 0]], index=df3.index)

valid_dataset = valid_dataset.append(df3)



myList = [file.split("/", 1)[1] for file in glob.glob('dataset/validation/drawings/*')]

df4 = pd.DataFrame(data=myList, columns=['filename'])

df4[categories] = pd.DataFrame([[0, 0, 0, 1, 0]], index=df4.index)

valid_dataset = valid_dataset.append(df3)



myList = [file.split("/", 1)[1] for file in glob.glob('dataset/validation/painting/*')]

df5 = pd.DataFrame(data=myList, columns=['filename'])

df5[categories] = pd.DataFrame([[0, 0, 0, 0, 1]], index=df5.index)

valid_dataset = valid_dataset.append(df5)
valid_dataset.reset_index(drop=True, inplace=True)

valid_dataset.tail()
valid_dataset.shape
test_dataset = pd.DataFrame(columns=[categories])

test_dataset.insert(loc=0, column='filename', value=None)

test_dataset.head()
del df1, df2, df3, df4, df5



df1 = test_dataset.copy()

df2 = test_dataset.copy()

df3 = test_dataset.copy()

df4 = test_dataset.copy()

df5 = test_dataset.copy()

df5.head()
myList = [file.split("/", 1)[1] for file in glob.glob('dataset/test/sculpture/*')]

test_dataset = pd.DataFrame(data=myList, columns=['filename'])

test_dataset[categories] = pd.DataFrame([[1, 0, 0, 0, 0]], index=dataset.index)



myList = [file.split("/", 1)[1] for file in glob.glob('dataset/test/iconography/*')]

df2 = pd.DataFrame(data=myList, columns=['filename'])

df2[categories] = pd.DataFrame([[0, 1, 0, 0, 0]], index=df2.index)

test_dataset = test_dataset.append(df2)



myList = [file.split("/", 1)[1] for file in glob.glob('dataset/test/engraving/*')]

df3 = pd.DataFrame(data=myList, columns=['filename'])

df3[categories] = pd.DataFrame([[0, 0, 1, 0, 0]], index=df3.index)

test_dataset = test_dataset.append(df3)



myList = [file.split("/", 1)[1] for file in glob.glob('dataset/test/drawings/*')]

df4 = pd.DataFrame(data=myList, columns=['filename'])

df4[categories] = pd.DataFrame([[0, 0, 0, 1, 0]], index=df4.index)

test_dataset = test_dataset.append(df3)



myList = [file.split("/", 1)[1] for file in glob.glob('dataset/test/painting/*')]

df5 = pd.DataFrame(data=myList, columns=['filename'])

df5[categories] = pd.DataFrame([[0, 0, 0, 0, 1]], index=df5.index)

test_dataset = test_dataset.append(df5)
test_dataset.reset_index(drop=True, inplace=True)

test_dataset.tail()
test_dataset.shape
datagen=ImageDataGenerator(rescale=1./255.)



train_generator=datagen.flow_from_dataframe(

    dataframe=dataset,

    directory="dataset/",

    x_col="filename",

    y_col=categories,

    batch_size=32,

    seed=42,

    shuffle=True,

    class_mode="other",

    target_size=(100,100))



valid_generator=datagen.flow_from_dataframe(

    dataframe=valid_dataset,

    directory="dataset/",

    x_col="filename",

    y_col=categories,

    batch_size=32,

    seed=42,

    shuffle=True,

    class_mode="other",

    target_size=(100,100))



test_generator=datagen.flow_from_dataframe(

    dataframe=test_dataset,

    directory="dataset/",

    x_col="filename",

    batch_size=1,

    seed=42,

    shuffle=False,

    class_mode=None,

    target_size=(100,100))
# model = Sequential([

#     Conv2D(32, (3, 3), padding='same', input_shape=(100,100,3), activation='relu'),

#     Conv2D(32, (3, 3), activation='relu'),

#     MaxPooling2D(pool_size=(2, 2)),

    

#     Dropout(0.25),

#     Conv2D(64, (3, 3), padding='same', activation='relu'),

#     Conv2D(64, (3, 3), activation='relu'),

#     MaxPooling2D(pool_size=(2, 2)),

    

#     Dropout(0.25),

#     Flatten(),

#     Dense(512, activation='relu'),

#     Dropout(0.5),

#     Dense(5, activation='sigmoid')

# ])

model = Sequential([

    Conv2D(32, (3, 3), input_shape=(100,100,3), activation='relu'),

    MaxPooling2D(pool_size=(2, 2)),

    Dropout(0.2),

    

    Conv2D(32, (3, 3), activation='relu'),

    MaxPooling2D(pool_size=(2, 2)),

    Dropout(0.2),

    

    Conv2D(64, (3, 3), activation='relu'),

    MaxPooling2D(pool_size=(2, 2)),

    Dropout(0.2),

    

    Flatten(),

    Dense(128, activation='relu'),

    Dropout(0.2),

    Dense(64, activation='relu'),

    Dropout(0.2),

    Dense(5, activation='softmax')

])



model.compile(optimizers.rmsprop(lr=0.0001, decay=1e-6),

              loss="binary_crossentropy", metrics=["accuracy"])
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size

STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size

STEP_SIZE_TEST=test_generator.n//test_generator.batch_size



model.fit_generator(generator=train_generator,

                    steps_per_epoch=STEP_SIZE_TRAIN,

                    validation_data=valid_generator,

                    validation_steps=STEP_SIZE_VALID,

                    epochs=50

)
test_generator.reset()

pred = model.predict_generator(test_generator, steps=STEP_SIZE_TEST, verbose=1)
pred_bool = (pred >0.5)
predictions = pred_bool.astype(int)



results = pd.DataFrame(predictions, columns=categories)

results["filename"] = test_generator.filenames

ordered_cols = ["filename"] + categories



# To get the same column order

results = results[ordered_cols]

results.to_csv("results.csv", index=False)
results.head()
input_ = Input(shape = (100,100,3))

x = Conv2D(32, (3, 3), padding = 'same')(input_)

x = Activation('relu')(x)

x = Conv2D(32, (3, 3))(x)

x = Activation('relu')(x)

x = MaxPooling2D(pool_size = (2, 2))(x)

x = Dropout(0.25)(x)

x = Conv2D(64, (3, 3), padding = 'same')(x)

x = Activation('relu')(x)

x = Conv2D(64, (3, 3))(x)

x = Activation('relu')(x)

x = MaxPooling2D(pool_size = (2, 2))(x)

x = Dropout(0.25)(x)

x = Flatten()(x)

x = Dense(512)(x)

x = Activation('relu')(x)

x = Dropout(0.5)(x)

output1 = Dense(1, activation = 'sigmoid')(x)

output2 = Dense(1, activation = 'sigmoid')(x)

output3 = Dense(1, activation = 'sigmoid')(x)

output4 = Dense(1, activation = 'sigmoid')(x)

output5 = Dense(1, activation = 'sigmoid')(x)



model = Model(input_, [output1, output2, output3, output4, output5])



model.compile(optimizers.rmsprop(lr = 0.0001, decay = 1e-6), 

              loss = ["binary_crossentropy", "binary_crossentropy", "binary_crossentropy",

                      "binary_crossentropy", "binary_crossentropy"], metrics = ["accuracy"])
def generator_wrapper(generator):

    for batch_x, batch_y in generator:

        yield (batch_x, [batch_y[:,i] for i in range(5)])
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size

STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size

STEP_SIZE_TEST=test_generator.n//test_generator.batch_size



model.fit_generator(generator=generator_wrapper(train_generator),

                    steps_per_epoch=STEP_SIZE_TRAIN,

                    validation_data=generator_wrapper(valid_generator),

                    validation_steps=STEP_SIZE_VALID,

                    epochs=1, verbose=2)
test_generator.reset()

pred = model.predict_generator(test_generator, steps=STEP_SIZE_TEST, verbose=1)
pred
print(type(pred[0]))