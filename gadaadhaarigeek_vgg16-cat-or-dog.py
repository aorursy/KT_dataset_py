# Importing the libraries 



import pandas as pd

import numpy as np 



import matplotlib.pyplot as plt

import seaborn as sns



from keras.models import Sequential, Model

from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, GlobalMaxPooling2D, Dropout

from keras.applications import VGG16

from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator, load_img

from keras import layers



from sklearn.model_selection import train_test_split





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
!rm -rf "./train"

!unzip "/kaggle/input/dogs-vs-cats/train.zip"
# Preparing the data

filenames = os.listdir("./train")

categories = []

for filename in filenames:

    category = filename.split('.')[0]

    if category == 'dog':

        categories.append(1)

    else:

        categories.append(0)



df = pd.DataFrame({

    'filename': filenames,

    'category': categories

})
# Looking at the data in dataframe

df.head()
# Changin the category to strings 

df["category"] = df["category"].replace({0: 'cat', 1: 'dog'}) 
# Shape of data

df.shape
# Distribution of count classes

sns.set(style="white")

sns.countplot(df["category"])
sample = np.random.choice(filenames)

image = load_img("./train/" + sample)

# Each image is of different shapes and has 3 channel for RGB

plt.imshow(image)

plt.show()
# Splitting the data

train_df, test_df = train_test_split(df, test_size=.2, stratify=df["category"])

train_df = train_df.reset_index()

test_df = test_df.reset_index()
aug_data = ImageDataGenerator(

    rotation_range = 15, 

    shear_range = .2, 

    zoom_range = .2, 

    horizontal_flip = True, 

    vertical_flip = True, 

    width_shift_range = 0.15, 

    height_shift_range = .15, 

    fill_mode = "nearest"

)
input_shape = (224, 224)

batch_size = 32



train_generator = aug_data.flow_from_dataframe(

    dataframe=train_df, 

    directory="./train/", 

    x_col = "filename",

    y_col="category", 

    class_mode = "categorical", 

    target_size = input_shape, 

    batch_size = batch_size

    )
test_generator = aug_data.flow_from_dataframe(

    dataframe=test_df, 

    directory="./train/", 

    x_col = "filename",

    y_col="category", 

    class_mode = "categorical", 

    target_size = input_shape, 

    batch_size = batch_size

    )
# Load the VGG16 model

# include_top = False means we don't want to take last 3 fully connected layers of VGG16 and we want weights trained on ImageNet data.

pre_trained_model = VGG16(include_top=False, weights="imagenet")
pre_trained_model.summary()
# We will fix the initial layers and only train the last stacked set of convolution layer

for layer in pre_trained_model.layers[:15]:

    layer.trainable = False

    

for layer in pre_trained_model.layers[15:]:

    layer.trainable = True
# Take the output of last pooling layer

last_pooling_layer = pre_trained_model.get_layer("block5_pool")

last_output = last_pooling_layer.output
# Flatten the output layer which has 512 units

x = GlobalMaxPooling2D()(last_output)



# After this add a fully connected layer with 512 units 

x = Dense(units=512, activation="relu")(x)



# Add a dropout 

x = Dropout(rate=.25)(x)



# Add a final layer for classification 

x = layers.Dense(units=2, activation="softmax")(x)



# Combine the model with our layers

model = Model(inputs=pre_trained_model.input, outputs=x)
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-4), metrics=['accuracy'])
batch_size = 32

epochs = 10

train_size = train_df.shape[0]
hist = model.fit_generator(generator = train_generator, epochs = epochs, validation_data = test_generator, verbose=1, 

                           steps_per_epoch = np.ceil(train_size/batch_size))
sns.set(style="whitegrid")

plt.figure(figsize=(20, 6))

plt.subplot(1, 2, 1)

plt.plot(hist.history['accuracy'])

plt.plot(hist.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='best')



plt.subplot(1, 2, 2)

plt.plot(hist.history['loss'])

plt.plot(hist.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='best')

plt.show()