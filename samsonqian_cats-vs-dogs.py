import tensorflow as tf

import keras



import pandas as pd

import numpy as np



import cv2



from matplotlib import pyplot as plt

from matplotlib import image as mpimg

%matplotlib inline

import seaborn as sns



import os

print(os.listdir("../input"))
main_dir = "../input"

train_dir = "train/train"

path = os.path.join(main_dir, train_dir)



for p in os.listdir(path):

    category = p.split(".")[0]

    img_array = cv2.imread(os.path.join(path,p),cv2.IMREAD_GRAYSCALE)

    new_img_array = cv2.resize(img_array, dsize=(80, 80))

    plt.imshow(new_img_array,cmap="gray")

    break
X = []

y = []

convert = lambda category : int(category == 'dog')

def create_test_data(path):

    for p in os.listdir(path):

        category = p.split(".")[0]

        category = convert(category)

        img_array = cv2.imread(os.path.join(path,p),cv2.IMREAD_GRAYSCALE)

        new_img_array = cv2.resize(img_array, dsize=(80, 80))

        X.append(new_img_array)

        y.append(category)
create_test_data(path)

X = np.array(X).reshape(-1, 80,80,1)

y = np.array(y)
X = X/255.0
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=X.shape[1:]),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),

    tf.keras.layers.MaxPooling2D(2, 2), 

    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"), 

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Flatten(), 

    tf.keras.layers.Dense(512, activation="relu"), 

    tf.keras.layers.Dense(1, activation="sigmoid")  

])
model.summary()
from tensorflow.keras.optimizers import RMSprop



model.compile(optimizer=RMSprop(lr=0.001),

              loss="binary_crossentropy",

              metrics = ["acc"])
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(rescale=1.0/255.0)

test_datagen  = ImageDataGenerator(rescale=1.0/255.0)



# --------------------

# Flow training images in batches of 20 using train_datagen generator

# --------------------

train_generator = train_datagen.flow_from_directory(train_dir,

                                                    batch_size=20,

                                                    class_mode="binary",

                                                    target_size=(150, 150))     

# --------------------

# Flow validation images in batches of 20 using test_datagen generator

# --------------------

validation_generator =  test_datagen.flow_from_directory(validation_dir,

                                                         batch_size=20,

                                                         class_mode="binary",

                                                         target_size=(150, 150))

"""
"""

history = model.fit_generator(train_generator,

                              validation_data=validation_generator,

                              steps_per_epoch=100,

                              epochs=15,

                              validation_steps=50,

                              verbose=2)

"""

history = model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)
#-----------------------------------------------------------

# Retrieve a list of list results on training and test data

# sets for each training epoch

#-----------------------------------------------------------

acc      = history.history[     'acc' ]

val_acc  = history.history[ 'val_acc' ]

loss     = history.history[    'loss' ]

val_loss = history.history['val_loss' ]



epochs   = range(len(acc)) # Get number of epochs



#------------------------------------------------

# Plot training and validation accuracy per epoch

#------------------------------------------------

plt.plot  ( epochs,     acc )

plt.plot  ( epochs, val_acc )

plt.title ('Training and validation accuracy')

plt.figure()



#------------------------------------------------

# Plot training and validation loss per epoch

#------------------------------------------------

plt.plot  ( epochs,     loss )

plt.plot  ( epochs, val_loss )

plt.title ('Training and validation loss'   )
train_dir = "test1/test1"

path = os.path.join(main_dir,train_dir)

#os.listdir(path)



X_test = []

id_line = []

def create_test1_data(path):

    for p in os.listdir(path):

        id_line.append(p.split(".")[0])

        img_array = cv2.imread(os.path.join(path,p),cv2.IMREAD_GRAYSCALE)

        new_img_array = cv2.resize(img_array, dsize=(80, 80))

        X_test.append(new_img_array)

create_test1_data(path)

X_test = np.array(X_test).reshape(-1,80,80,1)

X_test = X_test/255



predictions = model.predict(X_test)



predicted_val = [int(round(p[0])) for p in predictions]
submission_df = pd.DataFrame({'id':id_line, 'label':predicted_val})

submission_df.to_csv("submission.csv", index=False)