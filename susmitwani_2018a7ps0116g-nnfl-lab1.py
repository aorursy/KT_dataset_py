import skimage.io

import os

import csv

import random

import pandas as pd

import numpy as np

import tensorflow as tf

import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img

from sklearn.model_selection import train_test_split

from keras.callbacks import EarlyStopping



from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
filenames = os.listdir("../input/nnfl-lab-1/training/training")

uniq_cat = []

for name in filenames:

    nn = name.split('_')[0]

    if nn not in uniq_cat:

        if(nn=='chair' or nn=='kitchen' or nn=='knife' or nn=='saucepan'):

            uniq_cat.append(nn)

print(type(filenames))

print(uniq_cat)
filenames = (os.listdir("../input/nnfl-lab-1/training/training"))

categories = []

files = []

for filename in filenames:

    category = filename.split('_')[0]

    if category == 'chair':

        categories.append(0)

        files.append(filename)

    elif category == 'kitchen':

        categories.append(1)

        files.append(filename)

    elif category == 'knife':

        categories.append(2)

        files.append(filename)

    elif category == 'saucepan':

        categories.append(3)

        files.append(filename)



df_train = pd.DataFrame({

    'filename': files,

    'category': categories

})



df_train.head()
filenames = (os.listdir("../input/nnfl-lab-1/testing/testing"))

categories = []

files = []

for filename in filenames:

    categories.append(-1)

    files.append(filename)



df_test = pd.DataFrame({

    'filename': files,

    'category': categories

})



df_test.head()
df_train['category'] = df_train['category'].replace({0:'chair', 1:'kitchen', 2:'knife', 3:'saucepan'})

df_train['category'].value_counts().plot.bar()
sample = random.choice(os.listdir("../input/nnfl-lab-1/training/training"))

image = load_img("../input/nnfl-lab-1/training/training/"+sample)

print(sample)

print("Shape of image is: ", image.size)

plt.imshow(image)
X_train, X_val = train_test_split(df_train, test_size=0.2, stratify=df_train['category'], random_state = 42)
print(X_train.shape)

print(X_val.shape)
train_datagen = ImageDataGenerator(rescale = 1./255.,

                                  shear_range=0.3,

                                  zoom_range=0.3,

                                  horizontal_flip = True,

                                  fill_mode='nearest',

                                  height_shift_range=0.2,

                                  width_shift_range=0.2)

validation_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_dataframe(

    X_train, 

    '../input/nnfl-lab-1/training/training/', 

    x_col='filename',

    y_col='category',

    target_size=(216, 216),

    class_mode='sparse',

    batch_size=32,

    color_mode='grayscale'

)

validation_generator = validation_datagen.flow_from_dataframe(

    X_val, 

    '../input/nnfl-lab-1/training/training/', 

    x_col='filename',

    y_col='category',

    target_size=(216, 216),

    class_mode='sparse',

    batch_size=8,

    color_mode='grayscale'

)
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(64, (5,5), activation='relu', input_shape=(216,216,1)),

    tf.keras.layers.MaxPooling2D(2,2, padding='same'),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2, padding='same'),

    tf.keras.layers.Dropout(0.2),

    

    tf.keras.layers.Conv2D(64,(5,5), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2, padding='same'), 

    tf.keras.layers.Dropout(0.2),

    

    tf.keras.layers.Conv2D(128,(5,5), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2, padding='same'), 

    tf.keras.layers.Dropout(0.2),

    

    tf.keras.layers.Conv2D(256, (3,3), activation = 'relu'),

    tf.keras.layers.MaxPooling2D(2,2, padding='same'),

    tf.keras.layers.Dropout(0.2),

    

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.Dense(4, activation='softmax')]

    )

model.summary()
model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

history = model.fit_generator(train_generator, epochs=40, verbose=1, validation_data = validation_generator)
model.save_weights("model.h5")
# Plot the chart for accuracy and loss on both training and validation

%matplotlib inline

import matplotlib.pyplot as plt

acc = history.history['accuracy'][:-30]

val_acc = history.history['val_accuracy'][:-30]

loss = history.history['loss'][:-30]

val_loss = history.history['val_loss'][:-30]



epochs = range(len(acc))



plt.plot(epochs, acc, 'r', label='Training accuracy')

plt.plot(epochs, val_acc, 'b', label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend()

plt.figure()



plt.plot(epochs, loss, 'r', label='Training Loss')

plt.plot(epochs, val_loss, 'b', label='Validation Loss')

plt.title('Training and validation loss')

plt.legend()



plt.show()
test_datagen = ImageDataGenerator(rescale=1./255)



test_generator = test_datagen.flow_from_dataframe(

    df_test, 

    "../input/nnfl-lab-1/testing/testing/", 

    x_col='filename',

    y_col=None,

    class_mode=None,

    target_size=(216,216),

    batch_size=1,

    color_mode='grayscale',

    shuffle=False

)
y_pred = model.predict_generator(test_generator, steps = 1267, verbose=0)

print(len(y_pred))
df_test['category'] = np.argmax(y_pred, axis=-1)

df_test.head(10)
sample_test = df_test.head(18)

sample_test.head()

plt.figure(figsize=(12, 24))

for index, row in sample_test.iterrows():

    filename = row['filename']

    category = row['category']

    img = load_img("../input/nnfl-lab-1/testing/testing/"+filename, target_size=(256, 256))

    plt.subplot(6, 3, index+1)

    plt.imshow(img)

    plt.xlabel(filename + '(' + "{}".format(category) + ')' )

plt.tight_layout()

plt.show()
submission_df = df_test.copy()

submission_df['id'] = submission_df['filename']

submission_df['label'] = submission_df['category']

submission_df.drop(['filename', 'category'], axis=1, inplace=True)

submission_df.to_csv('submission.csv', index=False)

submission_df.head()
 