import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import numpy as np

from plotly import graph_objs as go

import pandas as pd

import os

import seaborn as sns

from tqdm import tqdm_notebook as tqdm
os.listdir("../input/")
import zipfile 

with zipfile.ZipFile("../input/"+"train"+".zip","r") as z:

    z.extractall("../kaggle/working/temp_unzip")
print(f"List of first ten image filenames: \n {os.listdir('../kaggle/working/temp_unzip/train')[:10]}")

print(f"Total number of images in training data: {len(os.listdir('../kaggle/working/temp_unzip/train'))}")
filenames = os.listdir('../kaggle/working/temp_unzip/train')

labels = [str(x)[:3] for x in filenames]

train_df = pd.DataFrame({'filename': filenames, 'label': labels})

train_df.head()
train_df['label'] = train_df['label'].map({'dog': '1', 'cat':'0'})

train_df.head()
print(f"The data has {train_df['label'].nunique()} unique classes")



for lab in train_df['label'].unique(): 

    #Subset to just that target 

    label_df = train_df[train_df['label']==lab].reset_index()

    cols = 5

    rows = 1

    fig = plt.figure(figsize = (4*cols - 1, 4.5*rows - 1))

    for c in range(cols):

        for r in range(rows):

            ax = fig.add_subplot(rows, cols, c*rows + r + 1)

            img = mpimg.imread('../kaggle/working/temp_unzip/train/'+label_df['filename'][c+r])

            ax.imshow(img)

            ax.set_title(str(label_df['filename'][c+r]))

    fig.suptitle(str(label_df['filename'][c+r][:3].upper()))

    plt.show()

    plt.close()
pd.DataFrame(train_df['label'].value_counts().reset_index())
dims_dict = {'image': [], 'width': [], 'height': [], 'channels': []}

for i in tqdm(range(len(train_df))):#['filename']):#train_pathlabel_df['image'].unique())):

    dims = mpimg.imread('../kaggle/working/temp_unzip/train/'+train_df['filename'][i]).shape

    dims_dict['image'].append(train_df['filename'][i])

    dims_dict['height'].append(dims[0])

    dims_dict['width'].append(dims[1])

    dims_dict['channels'].append(dims[2])



dims_df = pd.DataFrame(dims_dict)

dims_df.head()
sns.distplot(dims_df['height'])

plt.title('Distribution of image heights');

plt.show()
sns.distplot(dims_df['width'])

plt.title('Distribution of image widths');
dims_df['label'] = dims_df['image'].apply(lambda x: x[:3])

dims_df.head(3)
sns.distplot(dims_df[dims_df['label']=='dog']['height'], label='dog')

sns.distplot(dims_df[dims_df['label']=='cat']['height'], label='cat')

plt.title('Distribution of image heights between cats and dogs')

plt.legend();

plt.show()
sns.distplot(dims_df[dims_df['label']=='dog']['width'], label='dog')

sns.distplot(dims_df[dims_df['label']=='cat']['width'], label='cat')

plt.title('Distribution of image widths between cats and dogs')

plt.legend();

plt.show()
from keras import models

from keras import layers



network = models.Sequential()

network.add(layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (200, 200, 3), name="conv_1"))

network.add(layers.MaxPooling2D((2,2), name="maxpool_1"))

network.add(layers.Conv2D(64, (3,3), activation = 'relu', name="conv_2"))

network.add(layers.MaxPooling2D((2,2), name = "maxpool_2"))

network.add(layers.Conv2D(128, (3,3), activation = 'relu', name="conv_3"))

network.add(layers.MaxPooling2D((2,2), name = "maxpool_3"))

network.add(layers.Conv2D(128, (3,3), activation = 'relu', name="conv_4"))

network.add(layers.MaxPooling2D((2,2), name = "maxpool_4"))



network.add(layers.Flatten())

network.add(layers.Dense(512, activation = 'relu', name="dense_1"))

network.add(layers.Dense(1, activation = 'sigmoid', name="dense_2"))

network.summary()
network.compile(optimizer = 'adam',

               loss = 'binary_crossentropy',

               metrics = ['accuracy'])
from keras.preprocessing.image import ImageDataGenerator

# Instantiate an ImageDataGenerator with 30% validation data 

datagen = ImageDataGenerator(rescale = 1./255,

                             validation_split=0.3)



# Call the `flow_from_dataframe` method to create a generator for training data. 

# The input dataframe to this method needs to contain the image paths & target labels

train_data_gen = datagen.flow_from_dataframe(dataframe=train_df,

                                             directory='../kaggle/working/temp_unzip/train/',#Target directory

                                             x_col = 'filename',

                                             y_col = 'label',

                                             class_mode = 'binary',#Since we use binary crossentropy

                                             target_size=(200, 200),#All images will be resized to this

                                             color_mode='rgb',

                                             batch_size = 32,

                                             shuffle=True,

                                             seed=42,

                                             subset = 'training',#Just for train data generation

                                             validate_filenames=False)



# Repeat for validation data 

val_data_gen  = datagen.flow_from_dataframe(dataframe=train_df,

                                            directory='../kaggle/working/temp_unzip/train/',

                                            x_col = 'filename',

                                            y_col = 'label',

                                            class_mode = 'binary',

                                            target_size=(200, 200),#All images will be resized to this

                                            color_mode='rgb',

                                            batch_size = 32,

                                            shuffle=True,

                                            seed=42,

                                            subset = 'validation',#Just for train data generation

#                                             interpolation='bilinear',#Can try nearest as well. Need to read up on this

                                            validate_filenames=False)
for data_array, label_array in train_data_gen:

    print(f"Shape of train data batch data is {data_array.shape}")

    print(f"Shape of train data batch labels is {label_array.shape}")

    break # The generator has infinite yield, as endlessly iterates over batches. We need to break it manually
history = network.fit_generator(train_data_gen,

                                steps_per_epoch = 100,

                                epochs=30,

                                validation_data = val_data_gen,

                                validation_steps=50

                               )
network.save('cats_and_dogs_small_model.h5')
print(history.history.keys())

train_acc = history.history['acc']

val_acc = history.history['val_acc']

train_loss = history.history['loss']

val_loss = history.history['val_loss']

n_epochs = len(train_acc)

fig = plt.figure(figsize = (15,8))

fig.add_subplot(121)

plt.plot(range(n_epochs), train_acc, color = 'orange', label = "Train accuracy")

plt.plot(range(n_epochs), val_acc, color = 'blue', label = "Validation accuracy")

plt.legend();

fig.add_subplot(122)

plt.plot(range(n_epochs), train_loss, color = 'orange', label = "Train loss")

plt.plot(range(n_epochs), val_loss, color = 'blue', label = "Validation loss")

plt.legend();
from keras.preprocessing.image import ImageDataGenerator



# ImageDataGenerator for validation data 

train_datagen = ImageDataGenerator(rescale = 1./255,

                                   rotation_range = 45,

                                   width_shift_range = 0.2,

                                   height_shift_range = 0.2,

                                   shear_range = 0.2,

                                   zoom_range = 0.2)

val_datagen = ImageDataGenerator(rescale = 1./255)



# Call the `flow_from_dataframe` method to create a generator for training data. 

# The input dataframe to this method needs to contain the image paths & target labels



# We split the training and validation data because augmentations/transformations should not be applied to valdation data 

from sklearn.model_selection import train_test_split

augmented_mod_train_df, augmented_mod_val_df = train_test_split(train_df, test_size=0.2)
print(f"Split between classes in train data: \n {augmented_mod_train_df['label'].value_counts()*100 / augmented_mod_train_df.shape[0]}")

print(f"Split between classes in validation data: \n {augmented_mod_val_df['label'].value_counts()*100 / augmented_mod_val_df.shape[0]}")
train_data_gen = train_datagen.flow_from_dataframe(dataframe=augmented_mod_train_df,

                                             directory='../kaggle/working/temp_unzip/train/',#Target directory

                                             x_col = 'filename',

                                             y_col = 'label',

                                             class_mode = 'binary',#Since we use binary crossentropy

                                             target_size=(200, 200),#All images will be resized to this

                                             color_mode='rgb',

                                             batch_size = 32,

                                             shuffle=True,

                                             seed=42,

                                             validate_filenames=False)



# Repeat for validation data 

val_data_gen  = val_datagen.flow_from_dataframe(dataframe=augmented_mod_train_df,

                                            directory='../kaggle/working/temp_unzip/train/',

                                            x_col = 'filename',

                                            y_col = 'label',

                                            class_mode = 'binary',

                                            target_size=(200, 200),#All images will be resized to this

                                            color_mode='rgb',

                                            batch_size = 32,

                                            shuffle=True,

                                            seed=42,

                                            validate_filenames=False)
from keras import models

from keras import layers

from keras import optimizers



network = models.Sequential()

network.add(layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (200, 200, 3), name="conv_1"))

network.add(layers.MaxPooling2D((2,2), name="maxpool_1"))

network.add(layers.Conv2D(64, (3,3), activation = 'relu', name="conv_2"))

network.add(layers.MaxPooling2D((2,2), name = "maxpool_2"))

network.add(layers.Conv2D(128, (3,3), activation = 'relu', name="conv_3"))

network.add(layers.MaxPooling2D((2,2), name = "maxpool_3"))

network.add(layers.Conv2D(128, (3,3), activation = 'relu', name="conv_4"))

network.add(layers.MaxPooling2D((2,2), name = "maxpool_4"))



network.add(layers.Flatten())

network.add(layers.Dropout(0.2))

network.add(layers.Dense(512, activation = 'relu', name="dense_1"))

network.add(layers.Dense(1, activation = 'sigmoid', name="dense_2"))

network.summary()
network.compile(optimizer = optimizers.adam(lr=1e-4),

               loss = 'binary_crossentropy',

               metrics = ['accuracy'])



history = network.fit_generator(train_data_gen,

                                steps_per_epoch = 50,

                                epochs=50,

                                validation_data = val_data_gen,

                                validation_steps=50

                               )

# network.save('cats_and_dogs_augmented_data.h5')
train_acc = history.history['acc']

val_acc = history.history['val_acc']

train_loss = history.history['loss']

val_loss = history.history['val_loss']

n_epochs = len(train_acc)

fig = plt.figure(figsize = (15,8))

fig.add_subplot(121)

plt.plot(range(n_epochs), train_acc, color = 'orange', label = "Train accuracy")

plt.plot(range(n_epochs), val_acc, color = 'blue', label = "Validation accuracy")

plt.legend();

fig.add_subplot(122)

plt.plot(range(n_epochs), train_loss, color = 'orange', label = "Train loss")

plt.plot(range(n_epochs), val_loss, color = 'blue', label = "Validation loss")

plt.legend();
import zipfile 

with zipfile.ZipFile("../input/"+"test1"+".zip","r") as z:

    z.extractall("../kaggle/working/temp_test_unzip")
filenames = os.listdir('../kaggle/working/temp_test_unzip/test1')

test_df = pd.DataFrame({'filename': filenames})

print(test_df.shape)

test_df.head()
test_data_gen  = val_datagen.flow_from_dataframe(dataframe=test_df,

                                            directory='../kaggle/working/temp_test_unzip/test1/',

                                            x_col = 'filename',

                                            y_col = None,

                                            class_mode = None,

                                            target_size=(200, 200),#All images will be resized to this

                                            color_mode='rgb',

                                            batch_size = 64,

                                            shuffle=False)#,

#                                             seed=42,

#                                             validate_filenames=False)
yhat = network.predict_generator(test_data_gen, steps=np.ceil(test_df.shape[0]/64))

print(yhat.shape)
submission_df = test_df.copy()

submission_df['id'] = submission_df['filename'].str.split('.').str[0]

submission_df['label'] = yhat

submission_df['label'] = np.where(yhat>0.5, 1, 0)

submission_df[['id', 'label']].to_csv('submission.csv', index=False)

submission_df[['id', 'label']].head()