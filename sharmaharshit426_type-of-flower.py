#Import Statements



#To load data

import os



#tqdm is used to disply progress of loops

from tqdm import tqdm



#Numpy

import numpy as np



#Preprocessing

from keras.preprocessing.image import img_to_array

from keras.preprocessing.image import load_img



#Label encoder calss will be used tp encode labels

from sklearn.preprocessing import LabelEncoder



#to_categorical method will be used to convert class vectors to binary class matrix(one hot encoding)

from keras.utils import to_categorical



#for handling image data

import cv2



#Core NN imports

from keras.models import Sequential

from keras.layers import Conv2D

from keras.layers import Dense

from keras.layers import Flatten

from keras.layers import Softmax

from keras.layers import MaxPool2D



import matplotlib.pyplot as plt

%matplotlib inline



import random



from keras.models import save_model

from keras.models import load_model
#Defining the paths 

tulip_path = "../input/flowers-recognition/flowers/tulip"

rose_path = "../input/flowers-recognition/flowers/rose"

daisy_path = "../input/flowers-recognition/flowers/daisy"

sunflower_path = "../input/flowers-recognition/flowers/sunflower"

dandelion_path = "../input/flowers-recognition/flowers/dandelion"
#Setting the image size to 150*150

img_size = 200

X = []

Y = []



def make_data(X, Y, path, label):

    for img_name in tqdm(os.listdir(path)):

        if img_name.endswith('jpg'):

            img_path = os.path.join(path, img_name)

            img = load_img(img_path, target_size=(img_size, img_size))

            img_array = img_to_array(img)

            X.append(np.array(img_array))

            Y.append(label)

    return X, Y
#Loading our data

X, Y = make_data(X, Y, tulip_path, 'Tulip')

n_tulip_images = len(X)



X, Y = make_data(X, Y, rose_path, 'Rose')

n_rose_images = len(X) - (n_tulip_images)



X, Y = make_data(X, Y, daisy_path, 'Daisy')

n_daisy_images = len(X) - (n_tulip_images + n_rose_images)



X, Y = make_data(X, Y, dandelion_path, 'Dandelion')

n_dandelion_images = len(X) - (n_daisy_images + n_tulip_images + n_rose_images)



X, Y = make_data(X, Y, sunflower_path, 'Sunflower')

n_sunflower_images = len(X) - (n_daisy_images + n_tulip_images + n_rose_images + n_dandelion_images)



print(f' Tulip Images\t\t\t{ n_tulip_images }')

print(f' Rose Images\t\t\t{ n_rose_images }')

print(f' Daisy Images\t\t\t{ n_daisy_images }')

print(f' Dandelion Images\t\t{ n_dandelion_images }')

print(f' Sunflower Images\t\t{ n_sunflower_images }')
#Converting X and Y to a numpy array to feed it to the NN

X = np.array(X)

Y = np.array(Y)



#Normalizing X

X /= 255
#Converting our labels to one-hot matrix

print(f'Before fit Transform {Y}')

l_encoder = LabelEncoder()

Y = l_encoder.fit_transform(Y)

print(f'Before One-hot encoding \n{ repr(Y) }')

Y = to_categorical(Y, 5)

print(f'After One-hot encoding \n{ Y }')


model = Sequential()

model.add(Conv2D(filters = 64, kernel_size = 3, padding='same',

                 input_shape = (img_size, img_size, 3), activation='relu'))

model.add(MaxPool2D(pool_size = 2))



model.add(Conv2D(filters = 32, kernel_size = 3, padding='same', activation='relu'))

model.add(MaxPool2D(pool_size = 2))



model.add(Conv2D(filters = 16, kernel_size = 3, padding = 'same', activation = 'relu'))

model.add(MaxPool2D(pool_size = 2))



model.add(Flatten())



model.add(Dense(64, activation = 'relu'))



model.add(Dense(32, activation = 'relu'))



model.add(Dense(5, activation = 'softmax'))



model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X, Y, verbose=1, batch_size=256, epochs=50)


plt.plot(history.history['accuracy'])

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.title('Model Accuracy')

plt.figure()

plt.plot(history.history['loss'])

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.title('Model Loss')


random.seed(0)



fig, ax = plt.subplots(4,2, figsize=(14, 15))

for i in range(4):

    for j in range(2):

        num = random.randint(0, 3000)

        #print(num)

        x = X[num]

        #x = x/255

        x = x.reshape(1, img_size, img_size, 3)

        op = model.predict(x)

        #print(op)

        op = np.argmax(op)

        #print(op)

        op = l_encoder.inverse_transform([op])

        #print(op)

        ax[i][j].imshow(X[num], interpolation=None)

        ax[i][j].set_title(f'Original Label:{l_encoder.inverse_transform([np.argmax(Y[num])])}\nPredicted Label:{op}')

        plt.tight_layout()