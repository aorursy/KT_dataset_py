import numpy as np # linear algebra

import gc

import matplotlib.pyplot as plt

import cv2 as cv

import os

import seaborn as sns

from keras.utils import to_categorical

from keras.models import Sequential

from keras import layers

from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report
root_path = '/kaggle/input/pokemon-generation-one/dataset/dataset'

classes = os.listdir(root_path)

count=0

count_dict = {}

print(f'Total number of pokemons: {len(classes)}')

for pokemon in classes:

    dir_path = os.path.join(root_path, pokemon)

    count+=len(os.listdir(dir_path))

    count_dict[pokemon] = len(os.listdir(dir_path))

print(f'Total number of images: {count}')

fig = plt.figure(figsize = (25, 5))

sns.lineplot(x = list(count_dict.keys()), y = list(count_dict.values())).set_title('Number of images for each pokemon')

plt.xticks(rotation = 90)

plt.margins(x=0)

plt.show()
# sorted the list of pokemons with respect to number of their appearances in the datasett

sorted_list =  sorted(count_dict.items(), key=lambda item: item[1], reverse=True)

sorted_list
# pick the 50 most frequently appears

chosen_pokemon = sorted_list[0:50]

chosen_pokemon
#functions to augment more images

def generate_extra_two(img):

    return img[0:96, 0:96,:], img[4:100, 4:100, :]

def generate_extra_three(img):

    return generate_extra_two(img)[0], generate_extra_two(img)[1], img[2:98,2:98,:]
#generate data and labels

X = []

y= []

poke_label_dict = {}

i=0

for pokemon in chosen_pokemon:

    name = pokemon[0]

    poke_label_dict[i] = name

    print(name+ ': ' + str(i))

    dir_path = os.path.join(root_path, name)

    j=0

    if pokemon[1] < 70:

        for filename in os.listdir(dir_path):   

            try:

                file_path = os.path.join(dir_path, filename)

                img = cv.imread(file_path, 1)

                img = cv.resize(img, (100, 100))

                extra = generate_extra_three(img)

                for e in range(3):

                    X.append(extra[e])

                    y.append(i)

                gc.collect()

            except:

                j+=1

                print(str(j)+ " Broken file(s)")



    elif pokemon[1] < 100:

        for filename in os.listdir(dir_path):   

            try:

                file_path = os.path.join(dir_path, filename)

                img = cv.imread(file_path, 1)

                img = cv.resize(img, (100, 100))

                extra = generate_extra_two(img)

                for e in range(2):

                    X.append(extra[e])

                    y.append(i)

                gc.collect()

            except:

                j+=1

                print(str(j)+ " Broken file(s)")    

    else:

        for filename in os.listdir(dir_path):   

            try:

                file_path = os.path.join(dir_path, filename)

                img = cv.imread(file_path, 1)

                img = cv.resize(img, (96, 96))

                X.append(img)

                y.append(i)

                gc.collect()

            except:

                j+=1

                print(str(j)+ " Broken file(s)")

    i+=1    

    
#convert X to 4-dimensional tensor

X = np.array(X).reshape(-1, 96, 96, 3)

#normalize X

X = X/255.0

#convert y to one-hot-encoded form

y = to_categorical(np.array(y))
#check shape of X and Y

X.shape, y.shape
#check if X has been normailized

X.max(),X.min()
#split data into training and test data

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state=500,test_size=0.2)
#check shape of training and test data

X_train.shape, X_test.shape, y_train.shape, y_test.shape
#split data into training and validation data

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, shuffle=True, random_state=500,test_size=0.4)
#check shape of training and validation data

X_train.shape, X_val.shape, y_train.shape, y_val.shape
#check length of label dictionary

len(poke_label_dict)
#visulaize some example in training set and some example in validation set

import random

def vis_ex(X,y):

    plt.figure(1,figsize=(15, 10))

    for i in range(1,9):

        c = random.randint(0, 2000)

        img = X[c]

        plt.subplot(5,8,i)

        plt.imshow(img)

        plt.title(poke_label_dict[np.argmax(y[c])])
vis_ex(X_train, y_train)
vis_ex(X_val, y_val)
model= Sequential()

#Phase 1: 2 Conv-> Pooling block

model.add(layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), 

                       padding='valid' , input_shape=(96,96,3),activation='relu'))

model.add(layers.BatchNormalization(axis=-1))

model.add(layers.MaxPooling2D(pool_size=(3,3), strides=(2,2)))

model.add(layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding='same', activation='relu'))

model.add(layers.BatchNormalization(axis=-1))

model.add(layers.MaxPooling2D(pool_size=(3,3), strides=(2,2)))

#Phase 2: Convol Phase

model.add(layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))

model.add(layers.BatchNormalization(axis=-1))

model.add(layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))

model.add(layers.BatchNormalization(axis=-1))

model.add(layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))

model.add(layers.BatchNormalization(axis=-1))

model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2))) #modified from pool(3,3) to fit the input

#Phase 3: Fully-connected Phase: #modify the second FC layer from original paper due to low number of training examples 

model.add(layers.Flatten())

model.add(layers.Dropout(0.5))

model.add(layers.Dense(units=1024, activation='relu'))

model.add(layers.BatchNormalization())

model.add(layers.Dropout(0.5))

model.add(layers.Dense(units=512, activation='relu'))

model.add(layers.BatchNormalization())

model.add(layers.Dropout(0.5))

model.add(layers.Dense(units=y_train.shape[1], activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#create datagen during training and validation

train_datagen = ImageDataGenerator(horizontal_flip=True,

                                   rotation_range = 45, 

                                   width_shift_range = 0.15,  

                                   height_shift_range = 0.15)

test_datagen = ImageDataGenerator(horizontal_flip=True)

train_generator =  train_datagen.flow(X_train, y_train, batch_size=256, shuffle=True)

val_generator = test_datagen.flow(X_val, y_val, batch_size=256, shuffle=True)
history = model.fit_generator(train_generator, validation_data=val_generator, epochs=100, 

                              steps_per_epoch=len(train_generator),validation_steps= len(val_generator))
y_pred = model.predict(X_test)
print(classification_report(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)))
#plot some predictions:

def vis_pred(X):

    plt.figure(1,figsize=(15, 10))

    for i in range(1,9):

        c = random.randint(0, 1000)

        img = X[c]

        plt.subplot(5,8,i)

        plt.imshow(img)

        plt.title(poke_label_dict[np.argmax(model.predict(img.reshape(1,96,96,3)))])

vis_pred(X_test)
# serialize model to JSON

model_json = model.to_json()

with open("model_pkm.json", "w") as json_file:

    json_file.write(model_json)

# serialize weights to HDF5

model.save_weights("model_pkm.h5")

print("Saved model to disk")