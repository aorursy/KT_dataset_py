!pip install tensorflow-gpu==1.14.0

!pip install keras==2.2.5
import tensorflow as tf

import keras
import numpy as np

import pandas as pd

import os



from glob import glob

# Use either PIL or cv2. PIL reads in RGB, cv2 in BGR, but cv2 is faster

# from PIL import Image

import cv2

import gc

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split



import keras

from keras.utils.np_utils import to_categorical

from keras.models import Sequential,Model

from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D



# For generating more images

from keras.preprocessing.image import ImageDataGenerator

# For reducing the learning rate when there is no increase in accuracy

from keras.callbacks import ReduceLROnPlateau 



from sklearn.metrics import confusion_matrix

from sklearn.utils.multiclass import unique_labels



# We will train on an already trained network for bigger accuracy.

# On a new network, the accuracy is about 67%. On MobileNetV2 it is about 80%

from keras.applications.mobilenet_v2 import MobileNetV2 





IMG_SIZE = (128,128)

TEST_RATIO = 0.20

NUM_CLASSES = 7
base_skin_dir = os.path.join('..', 'input/skin-cancer-mnist-ham10000')



# Search for every image in the file.

imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x

                     for x in glob(os.path.join(base_skin_dir, '*', '*.jpg'))}

print(imageid_path_dict["ISIC_0027419"])

# We need the lesion for creating a map of the diseases.

type_dict = {

    'nv': 'Melanocytic nevi',

    'mel': 'Melanoma',

    'bkl': 'Benign keratosis-like lesions ',

    'bcc': 'Basal cell carcinoma',

    'akiec': 'Actinic keratoses',

    'vasc': 'Vascular lesions',

    'df': 'Dermatofibroma'

}

names = ["nv","mel","bkl","bcc","akiec","vasc","df"]

df = pd.read_csv(os.path.join(base_skin_dir, 'HAM10000_metadata.csv'))



df['path'] = df['image_id'].map(imageid_path_dict.get)

df['cell_type_idx'] = pd.Categorical(df['dx']).codes

df.head()
df['image'] = df['path'].map(lambda x: np.asarray((cv2.resize(cv2.imread(x),IMG_SIZE))))
X = np.asarray(df['image'].tolist()) # Getting only the image from the dataframe

y = to_categorical(df['cell_type_idx'], num_classes = NUM_CLASSES) # One hot encode the output



df = None

del df

gc.collect()
print("First pixel before: ",X[0][0][0])

X = X/255.0

print("First pixel after: ",X[0][0][0])
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = TEST_RATIO, shuffle = True, random_state = 12345)

X = None

y = None

del X

del y

gc.collect()
print("Number of training images: ",x_train.shape[0])

print("Number of testing images: ",x_test.shape[0])
def create_model(X,y,X_tst,y_tst,epochs = 30,batch_size = 12):

    # Creating a generator for generating multiple images

    

    datagen = ImageDataGenerator(featurewise_center=False,  # set input mean to 0 over the dataset

                                 samplewise_center=False,  # set each sample mean to 0

                                 featurewise_std_normalization=False,  # divide inputs by std of the dataset

                                 samplewise_std_normalization=False,  # divide each input by its std

                                 zca_whitening=False,  # apply ZCA whitening

                                 rotation_range=45,  # randomly rotate images in the range (degrees, 0 to 180)

                                 zoom_range = 0.1, # Randomly zoom image 

                                 width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

                                 height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

                                 horizontal_flip=True,  # randomly flip images

                                 vertical_flip=True)  # randomly flip images

    

    datagen.fit(X)

    

    model = MobileNetV2(weights='imagenet',include_top = False,input_shape = (128,128,3))



    x = model.output #Take the last layer

    x = GlobalAveragePooling2D()(x) #Add a GlobalAvgPooling        

    x = Dense(1024, activation='relu')(x)



    out = Dense(NUM_CLASSES, activation='softmax')(x)



    model = Model(inputs=model.input, outputs=out)        



    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 

                                                patience=3, 

                                                verbose=1, 

                                                factor=0.5, 

                                                min_lr=0.000005)

    

    history = model.fit_generator(datagen.flow(X,y, batch_size=batch_size),

                                  steps_per_epoch=X.shape[0] // batch_size,

                                  epochs=epochs,

                                  validation_data=(X_tst,y_tst),

                                  validation_steps=X_tst.shape[0] // batch_size,

                                  callbacks=[learning_rate_reduction]

)

    return model,history
model,model_history = create_model(x_train,y_train,x_test,y_test)

model.save("disease.h5")