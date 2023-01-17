# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import csv

import os

csv_file = os.listdir("../input")

print(csv_file)



# Any results you write to the current directory are saved as output.
# Read input file



features = []

names = []

with open('../input/musicfeatures/data.csv','rt')as f:

    data = csv.reader(f)

    for row in data:

        features.append(np.array(row[1:]))

features = np.array(features)



key = features[0]

features = features[1:,:]

labels = features[:,-1]

categories = list(set(labels))



features = features[:,:-1]

key = key[:-1]



print('Features -', features.shape, key)

print('Categories -', categories, len(categories))



labels_temp = []

for i in range(len(categories)):

    for label in labels:

        if label == categories[i]:

            labels_temp.append(i)

labels = np.array(labels_temp)
# Convert string features to number

features = features.astype(np.float)

print(features[0])
# PCA

from sklearn.decomposition import PCA



features_copy = features.copy()

for i in range(len(key)):

    Xi = features[:,i]

    features_copy[:,i] = (Xi-np.mean(Xi))/np.std(Xi)

    

pca = PCA(n_components = 22)

PC = pca.fit_transform(features_copy)

print(pca.explained_variance_ratio_)

cumulated = []

cumulated_score = 0

for val in pca.explained_variance_ratio_:

    cumulated_score += val

    cumulated.append(cumulated_score)

print(cumulated)

print(PC.shape)
from sklearn.utils import shuffle



X, Y = shuffle(PC, labels)

print(X.shape, Y.shape)



train_split = 1



X_Train = X[:int(1000*train_split), :]

Y_Train = Y[:int(1000*train_split)]

X_Val = X[int(1000*train_split):, :]

Y_Val = Y[int(1000*train_split):]



print(X_Train.shape, Y_Train.shape, X_Val.shape, Y_Val.shape)
from keras.layers import Input, Dense, BatchNormalization, Activation, Dropout 

from keras.models import Model

from keras import optimizers





from numpy.random import seed

seed(2019)



from tensorflow import set_random_seed

set_random_seed(2019)



inputs = Input(shape=(22,))



x = Dense(32)(inputs)

x = BatchNormalization()(x)

#x = Dropout(0.5)(x)

x = Activation('relu')(x)



x = Dense(64)(x)

x = BatchNormalization()(x)

#x = Dropout(0.5)(x)

x = Activation('relu')(x)



x = Dense(64)(x)

x = BatchNormalization()(x)

x = Dropout(0.5)(x)

x = Activation('relu')(x)



x = Dense(32)(x)

x = BatchNormalization()(x)

#x = Dropout(0.5)(x)

x = Activation('relu')(x)



predictions = Dense(10, activation='softmax')(x)



model = Model(inputs=inputs, outputs=predictions)

opt = optimizers.SGD(lr=0.01)



model.compile(optimizer=opt,

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])



history = model.fit(np.array(X_Train), np.array(Y_Train), batch_size = 32, epochs = 250, validation_split = 0.2)
from matplotlib import pyplot as plt



plt.plot(history.history['acc'])

plt.plot(history.history['loss'])

plt.title('Training')

plt.ylabel('Score')

plt.xlabel('Epoch')

plt.legend(['Accuracy', 'Loss'], loc='upper left')

plt.show()
from matplotlib import pyplot as plt



plt.plot(history.history['val_acc'])

plt.plot(history.history['val_loss'])

plt.title('Validation')

plt.ylabel('Score')

plt.xlabel('Epoch')

plt.legend(['Accuracy', 'Loss'], loc='upper left')

plt.show()
# Read test data



features = []

names = []

offsets = []

with open('../input/test-songs/features.csv','rt')as f:

    data = csv.reader(f)

    for row in data:

        features.append(np.array(row))

        names.append(row[0])

        offsets.append(row[-1])

        

features = np.array(features)

features = features[1:,1:-1].astype(np.float)

names = np.array(names[1:])

offsets = np.array(offsets[1:])



print('Test Data Shape -', features.shape)
# Predict on features with trained model

from scipy import stats



features_copy = features.copy()

for i in range(len(key)):

    Xi = features[:,i]

    features_copy[:,i] = (Xi-np.mean(Xi))/np.std(Xi)



song_names = list(set(names))

song_groups = {song_name:[] for song_name in song_names}

song_pred = {song_name:[] for song_name in song_names}

song_genre = {song_name:[] for song_name in song_names}

pred_count = {song_name:{category:0 for category in categories} for song_name in song_names}



for i in range(len(names)):

    song_groups[names[i]].append(features_copy[i])



for song_name in song_names:

    features = song_groups[song_name]

    for feature in features:

        feature = pca.transform(np.reshape(feature, (1, feature.shape[0])))

        pred = model.predict(feature)

        idx = np.argmax(pred)

        final_pred = categories[idx]

        song_pred[song_name].append(final_pred)

    for pred_temp in song_pred[song_name]:

        pred_count[song_name][pred_temp] += 1



print(pred_count)
for song_name in song_names:

    stats = pred_count[song_name]

    song_genre[song_name] = max(stats, key=stats.get)

    

print(song_genre)