import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

%matplotlib inline

import matplotlib.pyplot as plt

from glob import glob

import seaborn as sns

from PIL import Image

from sklearn.model_selection import train_test_split

import keras

from keras.utils.np_utils import to_categorical

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D



from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

print("Import completed")
base_skin_dir = os.path.join('..', 'input/skin-cancer-mnist-ham10000')

imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x

                     for x in glob(os.path.join(base_skin_dir, '*', '*.jpg'))}



lesion_type_dict = {

    'nv': 'Melanocytic nevi',

    'mel': 'Melanoma',

    'bkl': 'Benign keratosis-like lesions ',

    'bcc': 'Basal cell carcinoma',

    'akiec': 'Actinic keratoses',

    'vasc': 'Vascular lesions',

    'df': 'Dermatofibroma'

}

print("Data dictionary")
tile_df = pd.read_csv(os.path.join(base_skin_dir, 'HAM10000_metadata.csv'))

tile_df['path'] = tile_df['image_id'].map(imageid_path_dict.get)

tile_df['cell_type'] = tile_df['dx'].map(lesion_type_dict.get) 

tile_df['cell_type_idx'] = pd.Categorical(tile_df['cell_type']).codes

tile_df.sample(5)
#filling null data age with default

tile_df['age'].fillna((tile_df['age'].mean()), inplace=True)



tile_df['image'] = tile_df['path'].map(lambda x: np.asarray(Image.open(x).resize((100,75))))

print("Images resized")
n_samples = 3

fig, m_axs = plt.subplots(7, n_samples, figsize = (4*n_samples, 3*7))

for n_axs, (type_name, type_rows) in zip(m_axs, tile_df.sort_values(['cell_type']).groupby('cell_type')):

    n_axs[0].set_title(type_name)

    for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(n_samples, random_state=1234).iterrows()):

        c_ax.imshow(c_row['image'])

        c_ax.axis('off')

fig.savefig('category_samples.png', dpi=300)
# Checking the image size distribution

tile_df['image'].map(lambda x: x.shape).value_counts()



#train

features=tile_df.drop(columns=['cell_type_idx'],axis=1)

target=tile_df['cell_type_idx']

x_train_o, x_test_o, y_train_o, y_test_o = train_test_split(features, target, test_size=0.20,random_state=1234)

print("Trained")
x_train = np.asarray(x_train_o['image'].tolist())

x_test = np.asarray(x_test_o['image'].tolist())



x_train_mean = np.mean(x_train)

x_train_std = np.std(x_train)



x_test_mean = np.mean(x_test)

x_test_std = np.std(x_test)



x_train = (x_train - x_train_mean)/x_train_std

x_test = (x_test - x_test_mean)/x_test_std

print("Normalized")



y_train = to_categorical(y_train_o, num_classes = 7)

y_test = to_categorical(y_test_o, num_classes = 7)

print("Classified")
x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size = 0.1, random_state = 2)



x_train = x_train.reshape(x_train.shape[0], *(75, 100, 3))

x_test = x_test.reshape(x_test.shape[0], *(75, 100, 3))

x_validate = x_validate.reshape(x_validate.shape[0], *(75, 100, 3))

print("Reshaped")
input_shape = (75, 100, 3)

num_classes = 7

model = Sequential([

    Conv2D(32, 3, padding='same', activation='relu', input_shape=input_shape),

    Conv2D(32, 3, padding='same', activation='relu'),

    MaxPooling2D(),

    Dropout(0.25),



    Conv2D(64, 3, padding='same', activation='relu', input_shape=input_shape),

    Conv2D(64, 3, padding='same', activation='relu'),

    MaxPooling2D(),

    Dropout(0.4),

    

    Conv2D(128, 3, padding='same', activation='relu'),

    MaxPooling2D(),

    Dropout(0.45),

    

    Flatten(),

    Dense(512, activation='relu'),

    Dropout(0.5),

    Dense(7, activation='softmax')

])

print("Modelled")
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(optimizer='adam',  loss='binary_crossentropy', metrics=['accuracy'])

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.00001)

model.summary()
## Fitting The model

datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images

datagen.fit(x_train)

print("fitting the model")

epochs = 5

batch_size = 10

history = model.fit_generator(

    datagen.flow(x_train,y_train, batch_size=batch_size),

    steps_per_epoch=x_train.shape[0] // batch_size,

    epochs=epochs,

    validation_data=(x_validate,y_validate),

    validation_steps=x_validate.shape[0] // batch_size

    ,callbacks=[learning_rate_reduction]

)

loss, accuracy = model.evaluate(x_test, y_test, verbose=1)

loss_v, accuracy_v = model.evaluate(x_validate, y_validate, verbose=1)

print("Validation: accuracy = %f  ;  loss_v = %f" % (accuracy_v, loss_v))

print("Test: accuracy = %f  ;  loss = %f" % (accuracy, loss))

model.save("my_model.h5")
#1. Function to plot model's validation loss and validation accuracy

def plot_model_history(model_history):

    fig, axs = plt.subplots(1,2,figsize=(15,5))

    # summarize history for accuracy

    axs[0].plot(range(1,len(model_history.history['accuracy'])+1),model_history.history['accuracy'])

    axs[0].plot(range(1,len(model_history.history['val_accuracy'])+1),model_history.history['val_accuracy'])

    axs[0].set_title('Model Accuracy')

    axs[0].set_ylabel('Accuracy')

    axs[0].set_xlabel('Epoch')

    axs[0].set_xticks(np.arange(1,len(model_history.history['accuracy'])+1),len(model_history.history['accuracy'])/10)

    axs[0].legend(['train', 'val'], loc='best')

    # summarize history for loss

    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])

    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])

    axs[1].set_title('Model Loss')

    axs[1].set_ylabel('Loss')

    axs[1].set_xlabel('Epoch')

    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)

    axs[1].legend(['train', 'val'], loc='best')

    plt.show()

plot_model_history(history)
from tensorflow.keras.models import load_model

load_model(

    filepath="my_model.h5",

    custom_objects=None,

    compile=True

)

print("Model loaded")
import tensorflow.compat.v1 as tf

converter = tf.lite.TFLiteConverter.from_keras_model_file('my_model.h5') 

tfmodel = converter.convert() 

open ("model.tflite" , "wb") .write(tfmodel)

print("Tensorflow lite model created")