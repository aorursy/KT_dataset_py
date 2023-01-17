import os

from glob import glob



import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding



from keras.preprocessing.image import ImageDataGenerator

from keras import layers

from keras import Model, Input



from keras.optimizers import Adam

from keras.callbacks import ReduceLROnPlateau, EarlyStopping



from sklearn.model_selection import train_test_split



%matplotlib inline

import matplotlib.pyplot as plt
skin_df = pd.read_csv("../input/HAM10000_metadata.csv")

skin_df.head()
BASE_DIR = "../input"



imageid_dict = {os.path.splitext(os.path.basename(x))[0]: x

                     for x in glob(os.path.join(BASE_DIR, '*', '*.jpg'))}



lesion_dict = {

    'nv': 'Melanocytic_nevi',

    'mel': 'melanoma',

    'bkl': 'Benign_keratosis-like_lesions',

    'bcc': 'Basal_cell_carcinoma',

    'akiec': 'Actinic_keratoses',

    'vasc': 'Vascular_lesions',

    'df': 'Dermatofibroma'

}



skin_df["path"] = skin_df["image_id"].map(imageid_dict.get) # map image_id to the path of that image

skin_df["cell_type"] = skin_df["dx"].map(lesion_dict.get) # map dx to type of lesion

skin_df["cell_type_idx"] = pd.Categorical(skin_df["cell_type"]).codes # give each cell type a category id

skin_df.head()
fig, ax1 = plt.subplots(1,1,figsize=(10,5))

skin_df["cell_type"].value_counts().plot(kind="bar", ax=ax1, title="Counts for each type of Lesions") # plot a graph counting the number of each cell type
from skimage.io import imread
skin_df["image"] = skin_df["path"].map(imread) # read the image to array values
# let's see what is the shape of each value in the image column

skin_df["image"].map(lambda x: x.shape).value_counts() 
from PIL import Image
reshaped_image = skin_df["path"].map(lambda x: np.asarray(Image.open(x).resize((64,64), resample=Image.LANCZOS).\

                                                          convert("RGB")).ravel())
out_vec = np.stack(reshaped_image, 0)
out_df = pd.DataFrame(out_vec)
out_df["label"] = skin_df["cell_type_idx"]
out_df.head()
import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding



from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras import layers

from tensorflow.keras import Model

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import ReduceLROnPlateau

%matplotlib inline



import matplotlib.pyplot as plt
skin_df = pd.read_csv('../input/hmnist_28_28_RGB.csv')
out_df.head()
X = out_df.drop("label", axis=1).as_matrix()

label = out_df["label"].values
X.shape, label.shape
X_mean = np.mean(X)

X_std = np.std(X)



X = (X - X_mean)/X_std
X_train_orig, X_test, y_train_orig, y_test = train_test_split(X, label, test_size=0.1,random_state=0)
X_train_orig.shape, X_test.shape, y_train_orig.shape, y_test.shape
X_train, X_val, y_train, y_val = train_test_split(X_train_orig, y_train_orig, test_size=0.2, random_state=1)
X_train.shape, X_val.shape, y_train.shape, y_val.shape
X_train = X_train.reshape(X_train.shape[0], *(64, 64, 3))

X_val = X_val.reshape(X_val.shape[0], *(64, 64, 3))

X_test = X_test.reshape(X_test.shape[0], *(64, 64, 3))
X_train.shape, X_val.shape, X_test.shape
y_train.shape
y_train = to_categorical(y_train)

y_val = to_categorical(y_val)

y_test = to_categorical(y_test)
y_train.shape, y_val.shape, y_test.shape
# Our input feature map is 64x64x3: 64x64 for the image pixels, and 3 for

# the three color channels: R, G, and B

img_input = layers.Input(shape=(64, 64, 3))



# First convolution extracts 16 filters that are 3x3

# Convolution is followed by max-pooling layer with a 2x2 window

x = layers.Conv2D(16, 3, activation='relu', padding='same')(img_input)

x = layers.MaxPooling2D(2)(x)



# Second convolution extracts 32 filters that are 3x3

# Convolution is followed by max-pooling layer with a 2x2 window

x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)

x = layers.MaxPooling2D(2)(x)



# Third convolution extracts 64 filters that are 3x3

# Convolution is followed by max-pooling layer with a 2x2 window

x = layers.Convolution2D(64, 3, activation='relu', padding='same')(x)

x = layers.MaxPooling2D(2)(x)



# Flatten feature map to a 1-dim tensor

x = layers.Flatten()(x)



# Create a fully connected layer with ReLU activation and 256 hidden units

x = layers.Dense(256, activation='relu')(x)



# Flatten feature map to a 1-dim tensor

x = layers.Flatten()(x)



# Add a dropout rate of 0.5

x = layers.Dropout(0.5)(x)



# Create a fully connected layer with ReLU activation and 512 hidden units

x = layers.Dense(512, activation='relu')(x)



# Add a dropout rate of 0.5

x = layers.Dropout(0.5)(x)



# Create output layer with a single node and sigmoid activation, since final classification is out of 7 types

output = layers.Dense(7, activation='softmax')(x)



# Configure and compile the model

model = Model(img_input, output)



optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)



model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.summary()
train_datagen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,

                             shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')



train_datagen.fit(X_train)



val_datagen = ImageDataGenerator()

val_datagen.fit(X_val)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
batch_size = 64

epochs = 30

history = model.fit_generator(train_datagen.flow(X_train,y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = val_datagen.flow(X_val, y_val),

                              verbose = 1, steps_per_epoch=(X_train.shape[0] // batch_size),

                              callbacks=[learning_rate_reduction])
loss_test, acc_test = model.evaluate(X_test, y_test, verbose=1)

loss_val, acc_val = model.evaluate(X_val, y_val, verbose=1)

print("Validation: accuracy = %f  ;  loss_v = %f" % (acc_val, loss_val))

print("Test: accuracy = %f  ;  loss = %f" % (acc_test, loss_test))

model.save("model.h5")
# Retrieve a list of accuracy results on training and test data

# sets for each training epoch

acc = history.history['acc']

val_acc = history.history['val_acc']



# Retrieve a list of list results on training and test data

# sets for each training epoch

loss = history.history['loss']

val_loss = history.history['val_loss']



# Get number of epochs

epochs = range(len(acc))



# Plot training and validation accuracy per epoch

plt.plot(epochs, acc)

plt.plot(epochs, val_acc)

plt.title('Training and validation accuracy')



plt.figure()



# Plot training and validation loss per epoch

plt.plot(epochs, loss)

plt.plot(epochs, val_loss)

plt.title('Training and validation loss')