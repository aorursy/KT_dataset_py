# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split

from keras.layers import Input, Flatten, Dense, Conv2D

from keras.models import Model

from keras.optimizers import Adam

from keras.utils import to_categorical



import os

import cv2

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Cell taken from https://www.kaggle.com/ibrahimsherify/lenet-using-keras-accuracy-99



# Create a basis for all imported objects

PATH = "../input/four-shapes/shapes/"

IMG_SIZE = 64 # Since 64x64 imgs

Shapes = ["circle", "square", "triangle", "star"]



# Create holding arrays for images

Labels = []

Dataset = []



# Parse through each folder and pull all images

for shape in Shapes:

    print("Getting data for", shape)

    #iterate through each file in the folder

    for path in os.listdir(PATH + shape):

        #add the image to the list of images

        img = cv2.imread(PATH + shape + '/' + path)

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        Dataset.append(img)

        #add an integer to the labels list 

        Labels.append(Shapes.index(shape))

        

# Print results

print("\nDataset Images size:", len(Dataset))

print("Image Shape:", Dataset[0].shape)

print("Labels size:", len(Labels))
# Normalize images

Dataset = np.array(Dataset)

Dataset = Dataset.astype("float32") / 255.0



# One hot encode labels (preventing any integer relation from forming)

Labels = np.array(Labels)

Labels = to_categorical(Labels)



# Split Dataset to train\test

(trainX, testX, trainY, testY) = train_test_split(Dataset, Labels, test_size=0.2, random_state=42)



print("X Train shape:", trainX.shape)

print("X Test shape:", testX.shape)

print("Y Train shape:", trainY.shape)

print("Y Test shape:", testY.shape)
# Define the input layer to accept images

input_layer = Input((64,64,3))



# Flatten the input for dense processing

x = Flatten()(input_layer)



# Compute some dense layer processing

x = Dense(300, activation = 'relu')(x)

x = Dense(200, activation = 'relu')(x)

x = Dense(150, activation = 'relu')(x)



# Output layer as an integer value

output_layer = Dense(len(Shapes), activation = 'softmax')(x)



model = Model(input_layer, output_layer)



model.summary()
# Declare the optimizer

opt = Adam(lr=0.0005)



# Compile the model

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])



# Train the model

model.fit(trainX

          , trainY

          , batch_size=32

          , epochs=10

          , shuffle=True)
# Now try to test this against the test data

model.evaluate(testX, testY)
# See visually the predictions made

CLASSES = np.array(Shapes)



preds = model.predict(testX)

preds_single = CLASSES[np.argmax(preds, axis = -1)]

actual_single = CLASSES[np.argmax(testY, axis = -1)]



n_to_show = 10

indices = np.random.choice(range(len(testX)), n_to_show)



fig = plt.figure(figsize=(15, 3))

fig.subplots_adjust(hspace=0.4, wspace=0.4)



for i, idx in enumerate(indices):

    img = testX[idx]

    ax = fig.add_subplot(1, n_to_show, i+1)

    ax.axis('off')

    ax.text(0.5, -0.35, 'pred = ' + str(preds_single[idx]), fontsize=10, ha='center', transform=ax.transAxes) 

    ax.text(0.5, -0.7, 'act = ' + str(actual_single[idx]), fontsize=10, ha='center', transform=ax.transAxes)

    ax.imshow(img)