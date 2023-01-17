import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from skimage.io import imread



from keras.applications import resnet50

from keras.optimizers import Adam

from keras.layers import GlobalAveragePooling2D, Dense

from keras.models import Model

from keras.callbacks import ModelCheckpoint
train = pd.read_csv('../input/aerial-cactus-identification/train.csv')

test = pd.read_csv('../input/aerial-cactus-identification/sample_submission.csv')

y_train = np.array(train.has_cactus)
X_train = []

for name in train.id:

    X_train.append(imread('../input/aerial-cactus-identification/train/train/'+name))
X_test = []

for name in test.id:

    X_test.append(imread('../input/aerial-cactus-identification/test/test/'+name))
X_train = np.array(X_train)

X_test = np.array(X_test)
#Preprocessing the inputs

X_train = resnet50.preprocess_input(X_train)

X_test = resnet50.preprocess_input(X_test)
#Retriving the model without the fully connected layers

base_model = resnet50.ResNet50(include_top=False, weights='imagenet')
#Let's add a GAP and a fully connected layer with relu function

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(1024, activation='relu')(x)

predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs = base_model.input, outputs = predictions)
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
#If have never trained this CNN, you can comment this line

model.load_weights('../input/weights/model.hdf5')
#Let's overfit in one batch

model.fit(X_train[:128], y_train[:128], epochs=30)
#Now we can train the model and save the weights in a .hdf5 file

checkpoint = ModelCheckpoint(filepath='model.hdf5')

model.fit(X_train, y_train, epochs=100, batch_size=128, callbacks=[checkpoint])
sample = pd.read_csv('../input/aerial-cactus-identification/sample_submission.csv')
sample.has_cactus = model.predict(X_test)
#Let's plot the distribution of the test targets

fig, ax = plt.subplots()

sns.kdeplot(sample.has_cactus, ax=ax)

ax.set_xlim([0,1])
sample.to_csv('sample_submission.csv', index=False)