# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import seaborn as sns



from sklearn import model_selection



from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score,auc, accuracy_score



from sklearn.preprocessing import StandardScaler, MinMaxScaler



from sklearn.linear_model import LogisticRegression



from sklearn.model_selection import train_test_split



from sklearn import datasets



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from keras.datasets import mnist



from keras.models import Sequential, load_model



from keras.layers import Dense, Dropout, Flatten



from keras.layers.convolutional import Conv2D, MaxPooling2D



from keras.utils.np_utils import to_categorical
import cv2

import os

import glob
img_dir = "/kaggle/input/chest-xray-pneumonia/chest_xray/train/NORMAL" # Enter Directory of all images 

data_path = os.path.join(img_dir,'*g')

files = glob.glob(data_path)

X_train=[]

y_train=[]

for f1 in files:

    img = cv2.imread(f1,cv2.IMREAD_GRAYSCALE) # Pour obtenir l'image grise

    img = cv2.resize(img, (100,100))

    img=img[:,:,np.newaxis]

    X_train.append(np.array(img))

    y_train.append(0)

n_train_normal = len(X_train)
img_dir = "/kaggle/input/chest-xray-pneumonia/chest_xray/test/NORMAL/" # Enter Directory of all images 

data_path = os.path.join(img_dir,'*g')

files = glob.glob(data_path)

X_test=[]

y_test=[]

for f1 in files:

    img = cv2.imread(f1,cv2.IMREAD_GRAYSCALE)

    img = cv2.resize(img, (100,100))

    img=img[:,:,np.newaxis]

    X_test.append(np.array(img))

    y_test.append(0)

n_test_normal = len(X_test)
plt.imshow(np.squeeze(X_train[10]),cmap="gray")

plt.title(y_train[10])
plt.imshow(np.squeeze(X_test[10]),cmap="gray")

plt.title(y_test[10])
np.array(X_train).shape
np.array(X_test).shape
plt.figure(figsize=(10,20))

for i in range(0,49) :

    plt.subplot(10,5,i+1)

    plt.axis('off')

    plt.imshow(np.squeeze(X_train[i]),cmap="gray")

    plt.title('Label: %i' % y_train[i])
plt.figure(figsize=(10,20))

for i in range(0,49) :

    plt.subplot(10,5,i+1)

    plt.axis('off')

    plt.imshow(np.squeeze(X_test[i]),cmap="gray")

    plt.title('Label: %i' % y_test[i])
img_dir = "/kaggle/input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA" # Enter Directory of all images 

data_path = os.path.join(img_dir,'*g')

files = glob.glob(data_path)

for f1 in files:

    img = cv2.imread(f1,cv2.IMREAD_GRAYSCALE)

    img = cv2.resize(img, (100,100))

    img=img[:,:,np.newaxis]    

    X_train.append(np.array(img))

    y_train.append(1)

n_train_pneu = len(X_train)-n_train_normal
img_dir = "/kaggle/input/chest-xray-pneumonia/chest_xray/test/PNEUMONIA" # Enter Directory of all images 

data_path = os.path.join(img_dir,'*g')

files = glob.glob(data_path)

for f1 in files:

    img = cv2.imread(f1,cv2.IMREAD_GRAYSCALE)

    img = cv2.resize(img, (100,100))

    img=img[:,:,np.newaxis]

    X_test.append(np.array(img))

    y_test.append(1)

n_test_pneu = len(X_test)-n_test_normal
print(n_train_pneu)

print(n_train_normal)

print(n_test_pneu)

print(n_test_normal)
plt.figure(figsize=(10,20))

for i in range(0,49) :

    plt.subplot(10,5,i+1)

    plt.axis('off')

    plt.imshow(np.squeeze(X_train[n_train_pneu+i]),cmap="gray")

    plt.title('Label: %i' % y_train[n_train_pneu+i])
plt.figure(figsize=(10,20))

for i in range(0,49) :

    plt.subplot(10,5,i+1)

    plt.axis('off')

    plt.imshow(np.squeeze(X_test[n_test_pneu+i]),cmap="gray")

    plt.title('Label: %i' % y_test[n_test_pneu+i])
plt.figure(figsize=(10,20))

for i in range(0,49) :

    plt.subplot(10,5,i+1)

    plt.axis('off')

    plt.imshow(np.squeeze(X_train[n_train_normal+i]),cmap="gray")

    plt.title('Label: %i' % y_train[n_train_normal+i])
plt.figure(figsize=(10,20))

for i in range(0,49) :

    plt.subplot(10,5,i+1)

    plt.axis('off')

    plt.imshow(np.squeeze(X_test[n_test_normal+i]),cmap="gray")

    plt.title('Label: %i' % y_test[n_test_normal+i])
# np.array(X_train).shape
filter(lambda x:x[1].shape==(100,100),enumerate(X_train))

[print(x) for x in filter(lambda x:x[1].shape==(100,100),enumerate(X_train))]
X_train = np.array(X_train)

y_train = np.array(y_train)
X_test = np.array(X_test)

y_test = np.array(y_test)
X_train.shape
X_test.shape
# Normalisation entre 0 et 1

X_train = X_train / 255

print(X_train[0][0])
# Normalisation entre 0 et 1

X_test = X_test / 255

print(X_test[0][0])
# Réseau convolutionnel simple

model = Sequential()

model.add(Conv2D(32, (5, 5), input_shape=(100, 100,1), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

# model.add(Dense(128, activation='relu'))

model.add(Dense(1))



# Compilation du modèle

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.summary()
# Apprentissage



train = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=200, verbose=1)

# train = model.fit(X_train_pro, y_train_pro, validation_data=(X_test_pro, y_test_pro), epochs=20, batch_size=200, verbose=1)
# Test

scores = model.evaluate(X_test, y_test, verbose=0)

print("Score : %.2f%%" % (scores[1]*100))
print(train.history['accuracy'])
print(train.history['val_accuracy'])
def plot_scores(train) :

    accuracy = train.history['accuracy']

    val_accuracy = train.history['val_accuracy']

    epochs = range(len(accuracy))

    plt.plot(epochs, accuracy, 'b', label='Score apprentissage')

    plt.plot(epochs, val_accuracy, 'r', label='Score validation')

    plt.title('Scores')

    plt.legend()

    plt.show()
plot_scores(train)
y_cnn = model.predict_classes(X_test)
cm = confusion_matrix(y_cnn,y_test)

print(cm)

plt.figure(figsize = (12,10))
plt.figure(figsize=(15,25))

n_test = X_test.shape[0]

i=1

for j in range(len(X_test)) :

    if (y_cnn[j] != y_test[j]) & (i<50):

        plt.subplot(10,5,i)

        plt.axis('off')

        plt.imshow(np.squeeze(X_test[j]),cmap="gray")

        pred_classe = y_cnn[j].argmax(axis=-1)

        plt.title('%d / %d' % (y_cnn[j], y_test[j]))

        i+=1
# Modèle CNN plus profond

model = Sequential()

# model.add(Conv2D(32, (3, 3), input_shape=(100, 100, 3), activation='relu'))

model.add(Conv2D(32, (3, 3), input_shape=(100, 100, 1), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))

# model.add(Conv2D(20, (3, 3), activation='relu'))



model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))



# model.add(Conv2D(20, (3, 3), activation='relu'))

model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))



model.add(Flatten())

model.add(Dense(1))



model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.summary()
# Apprentissage

train = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=200, verbose=1)



# Test

scores = model.evaluate(X_test, y_test, verbose=0)

print("Score : %.2f%%" % (scores[1]*100))
plot_scores(train)
model.save('pneu_cnn2.h5')
new_model = load_model('pneu_cnn2.h5')

new_model.summary()
scores = new_model.evaluate(X_test, y_test, verbose=0)

print("Score : %.2f%%" % (scores[1]*100))
y_cnn = model.predict_classes(X_test)



cm = confusion_matrix(y_cnn,y_test)

print(cm)
mg_dir = "/kaggle/input/chest-xray-pneumonia/chest_xray/train/NORMAL" # Enter Directory of all images 

data_path = os.path.join(img_dir,'*g')

files = glob.glob(data_path)

X_train=[]

y_train=[]

for f1 in files:

    img = cv2.imread(f1) # Pour obtenir l'image grise

    img = cv2.resize(img, (100,100))

    X_train.append(np.array(img))

    y_train.append(0)

n_train_normal = len(X_train)
img_dir = "/kaggle/input/chest-xray-pneumonia/chest_xray/test/NORMAL/" # Enter Directory of all images 

data_path = os.path.join(img_dir,'*g')

files = glob.glob(data_path)

X_test=[]

y_test=[]

for f1 in files:

    img = cv2.imread(f1)

    img = cv2.resize(img, (100,100))

    X_test.append(np.array(img))

    y_test.append(0)

n_test_normal = len(X_test)
img_dir = "/kaggle/input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA" # Enter Directory of all images 

data_path = os.path.join(img_dir,'*g')

files = glob.glob(data_path)

for f1 in files:

    img = cv2.imread(f1)

    img = cv2.resize(img, (100,100))

    X_train.append(np.array(img))

    y_train.append(1)

n_train_pneu = len(X_train)-n_train_normal
img_dir = "/kaggle/input/chest-xray-pneumonia/chest_xray/test/PNEUMONIA" # Enter Directory of all images 

data_path = os.path.join(img_dir,'*g')

files = glob.glob(data_path)

for f1 in files:

    img = cv2.imread(f1)

    img = cv2.resize(img, (100,100))

    X_test.append(np.array(img))

    y_test.append(1)

n_test_pneu = len(X_test)-n_test_normal
X_train = np.array(X_train)

y_train = np.array(y_train)
X_test = np.array(X_test)

y_test = np.array(y_test)
X_train = X_train / 255

X_test = X_test / 255
from keras.applications import VGG16
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(100,100,3)) ## Pour le première couche, les images d'entré n'ont pas les mêmes format avec le modèle

vgg16.trainable = False

vgg16.summary()
model = Sequential()

model.add(vgg16)

model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dense(128, activation='relu'))

model.add(Dense(1,activation="sigmoid"))
model.summary()


model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

train = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=200, verbose=1)
plot_scores(train)
# Test

scores = model.evaluate(X_test, y_test, verbose=0)

print("Score : %.2f%%" % (scores[1]*100))
for i in range (len(vgg16.layers)):

    print (i,vgg16.layers[i])
for layer in vgg16.layers[16:]:

    layer.trainable=True

for layer in vgg16.layers[0:15]:

    layer.trainable=False
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

train = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=200, verbose=1)
scores = model.evaluate(X_test, y_test, verbose=0)

print("Score : %.2f%%" % (scores[1]*100))
plot_scores(train)
y_cnn = model.predict_classes(X_test)
plt.figure(figsize=(15,25))

n_test = X_test.shape[0]

i=1

for j in range(len(X_test)) :

    if (y_cnn[j] != y_test[j]) & (i<50):

        plt.subplot(10,5,i)

        plt.axis('off')

        plt.imshow(X_test[j])

        pred_classe = y_cnn[j].argmax(axis=-1)

        plt.title('%d / %d' % (y_cnn[j], y_test[j]))

        i+=1
y_cnn = model.predict_classes(X_test)



cm = confusion_matrix(y_cnn,y_test)

print(cm)