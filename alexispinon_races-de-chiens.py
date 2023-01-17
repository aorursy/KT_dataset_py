# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from matplotlib import pyplot as plt

%matplotlib inline
df = pd.read_csv('../input/dog-breed-identification/labels.csv')

df.head()
df.describe()
df['breed'].value_counts()
from tensorflow.keras.preprocessing.image import load_img



images = []

for img in df['id'].values.tolist():

    #print(img)

    images.append(np.asarray(load_img('../input/dog-breed-identification/train/'+ img +'.jpg', target_size=(100, 100), color_mode='rgb')))
print(images[0])

print(images[0].shape)

plt.imshow(images[0]) #pas très beau avec le cmap grey...

plt.axis('off')

plt.title(df.breed[0])
df_list_breed = df['breed'].values.tolist()

races = set(df_list_breed) #set() récupère les valeurs uniques

# on map les races avec un chiffre pour pouvoir utiliser to_categorical (il y a peut être plus optimisé)

print(races)

races = ['siberian_husky', 'border_terrier', 'lhasa', 'japanese_spaniel', 'miniature_pinscher', 'boxer', 'rhodesian_ridgeback', 'gordon_setter', 'collie', 'miniature_poodle', 'otterhound', 'borzoi', 'irish_terrier', 'black-and-tan_coonhound', 'dandie_dinmont', 'german_shepherd', 'maltese_dog', 'tibetan_terrier', 'west_highland_white_terrier', 'miniature_schnauzer', 'norwegian_elkhound', 'beagle', 'english_springer', 'kuvasz', 'affenpinscher', 'silky_terrier', 'scotch_terrier', 'irish_wolfhound', 'yorkshire_terrier', 'boston_bull', 'saluki', 'whippet', 'standard_poodle', 'vizsla', 'keeshond', 'malinois', 'pembroke', 'toy_poodle', 'giant_schnauzer', 'staffordshire_bullterrier', 'walker_hound', 'american_staffordshire_terrier', 'labrador_retriever', 'toy_terrier', 'pomeranian', 'english_foxhound', 'irish_water_spaniel', 'malamute', 'greater_swiss_mountain_dog', 'german_short-haired_pointer', 'afghan_hound', 'irish_setter', 'flat-coated_retriever', 'dingo', 'papillon', 'italian_greyhound', 'english_setter', 'shih-tzu', 'basenji', 'briard', 'redbone', 'kerry_blue_terrier', 'wire-haired_fox_terrier', 'norwich_terrier', 'norfolk_terrier', 'weimaraner', 'great_pyrenees', 'lakeland_terrier', 'bouvier_des_flandres', 'bernese_mountain_dog', 'basset', 'schipperke', 'sealyham_terrier', 'cairn', 'newfoundland', 'shetland_sheepdog', 'brittany_spaniel', 'rottweiler', 'tibetan_mastiff', 'french_bulldog', 'border_collie', 'soft-coated_wheaten_terrier', 'bull_mastiff', 'chihuahua', 'sussex_spaniel', 'bloodhound', 'ibizan_hound', 'african_hunting_dog', 'australian_terrier', 'chesapeake_bay_retriever', 'leonberg', 'groenendael', 'blenheim_spaniel', 'great_dane', 'doberman', 'mexican_hairless', 'chow', 'komondor', 'brabancon_griffon', 'kelpie', 'bluetick', 'samoyed', 'cardigan', 'golden_retriever', 'curly-coated_retriever', 'clumber', 'cocker_spaniel', 'appenzeller', 'airedale', 'bedlington_terrier', 'scottish_deerhound', 'pug', 'eskimo_dog', 'entlebucher', 'dhole', 'saint_bernard', 'standard_schnauzer', 'old_english_sheepdog', 'welsh_springer_spaniel', 'pekinese']

i=0

map_race = {}

for race in races:

    map_race[race] = i

    i= i + 1
i = 0

y = []

for sample in df['breed']:

    y.append(map_race[sample])

print(y)
from keras.utils.np_utils import to_categorical



X = np.array(images)

y = y



print(y)

print(y.shape)
plt.figure(figsize=(10,20))

for i in range(0,49) :

    plt.subplot(10,5,i+1)

    plt.axis('off')

    plt.imshow(images[i])

    plt.title(df['breed'][i])
# Normalisation entre 0 et 1

X = X / 255

print(X[0][0])
def plot_scores(train) :

    accuracy = train.history['accuracy']

    val_accuracy = train.history['val_accuracy']

    epochs = range(len(accuracy))

    plt.plot(epochs, accuracy, 'b', label='Score apprentissage')

    plt.plot(epochs, val_accuracy, 'r', label='Score validation')

    plt.title('Scores')

    plt.legend()

    plt.show()
import cv2

import os

import glob
from sklearn.model_selection import train_test_split

from keras.models import Sequential, load_model

from keras.layers import Dense, Dropout, Flatten

from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.utils.np_utils import to_categorical



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



X_train1 = np.array(X_train)

X_test1 = np.array(X_test)

y_train1 = np.array(y_train)

y_test1 = np.array(y_test)



y_train1 = to_categorical(y_train1)

y_comparaison_finale = np.array(y_test1) 

y_test1 = to_categorical(y_test1)
X_train1.shape
model = Sequential()

model.add(Conv2D(32, (5, 5), input_shape=(100, 100, 3), activation='relu')) #attention dim image et b&w ou rgb

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dense(50, activation='relu'))

model.add(Dense(120)) #attention, 120 races de chiens



model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.summary()
print(X_train1.shape)

print(X_test1.shape)

print(y_train1.shape)

print(y_test1.shape)
# Apprentissage

train = model.fit(X_train1, y_train1, validation_data=(X_test1, y_test1), epochs=50, batch_size=200, verbose=1)



# Test

scores = model.evaluate(X_test1, y_test1, verbose=0)

print("Score : %.2f%%" % (scores[1]*100))
plot_scores(train)
from keras.applications import VGG16
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(100,100,3))

vgg16.trainable = False
vgg16.summary()
model = Sequential()

model.add(vgg16)

model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dense(120, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
train = model.fit(X_train1, y_train1, validation_data=(X_test1, y_test1), epochs=20, batch_size=200, verbose=1)
plot_scores(train)
for layer in vgg16.layers[10:]:

    layer.trainable=True

for layer in vgg16.layers[0:10]:

    layer.trainable=False
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

train = model.fit(X_train1, y_train1, validation_data=(X_test1, y_test1), epochs=20, batch_size=200, verbose=1)
plot_scores(train)
from keras.applications import InceptionV3, ResNet50V2
inceptionV3 = InceptionV3(weights='imagenet', include_top=False, input_shape=(100,100,3))
inceptionV3.summary()
model = Sequential()

model.add(inceptionV3)

model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dense(120, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

train = model.fit(X_train1, y_train1, validation_data=(X_test1, y_test1), epochs=20, batch_size=200, verbose=1)
plot_scores(train)
for layer in inceptionV3.layers[15:]:

    layer.trainable=True

for layer in inceptionV3.layers[0:15]:

    layer.trainable=False
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

train = model.fit(X_train1, y_train1, validation_data=(X_test1, y_test1), epochs=50, batch_size=200, verbose=1)
plot_scores(train)
y_hat = model.predict_classes(X_test1)
plt.figure(figsize=(10,20))

i=0

j=0



while i<20 :

    if  y_test[j] != y_hat[j] :

        plt.subplot(10,2,i+1)

        plt.axis('off')

        plt.imshow(X[j])

        plt.title(races[y_comparaison_finale[j]] + " / " + races[y_hat[j]])

        i += 1

    j+=1