# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import cv2

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from keras.preprocessing.image import load_img

img_name = 'IM-0122-0001.jpeg'

img_normal = load_img('/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/train/NORMAL/' + img_name)



print('NORMAL')

plt.imshow(img_normal)

plt.show()
img_name = 'person1007_virus_1690.jpeg'

img_pneumonia = load_img('/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/train/PNEUMONIA/' + img_name)



print('One of the PNEUMONIA')

plt.imshow(img_pneumonia)

plt.show()
Normal_train = os.listdir('/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/train/NORMAL')

Pneumonia_train = os.listdir('/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/train/PNEUMONIA')

from PIL import Image

image_arr_train =[]

labels_train = []



for img in Normal_train:

    try:

        image = cv2.imread('/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/train/NORMAL/' + img)

        image_from_array = Image.fromarray(image, 'RGB')  #Read in the image with PIL image function in colour mode.

        resize_img = image_from_array.resize((32, 32))  #Resize the image to 32 * 32

        image_arr_train.append(np.array(resize_img))

        labels_train.append(0)

        

    except AttributeError:

        print("An error occured while reading in the image")



for img in Pneumonia_train:

    try:

        image=cv2.imread('/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/train/PNEUMONIA/' + img)

        image_from_array = Image.fromarray(image, 'RGB')

        resize_img = image_from_array.resize((32, 32))

        image_arr_train.append(np.array(resize_img))

        labels_train.append(1)

        

    except AttributeError:

        print("An error occur while reading the image")
Normal_test = os.listdir('/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/test/NORMAL')

Pneumonia_test = os.listdir('/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/test/PNEUMONIA')
image_arr_test =[]

labels_test = []



for img in Normal_test:

    try:

        image = cv2.imread('/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/test/NORMAL/' + img)

        image_from_array = Image.fromarray(image, 'RGB')  #Read in the image with PIL image function in colour mode.

        resize_img = image_from_array.resize((32, 32))  #Resize the image to 32 * 32

        image_arr_test.append(np.array(resize_img))

        labels_test.append(0)

        

    except AttributeError:

        print("An error occured while reading in the image")



for img in Pneumonia_test:

    try:

        image=cv2.imread('/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/test/PNEUMONIA/' + img)

        image_from_array = Image.fromarray(image, 'RGB')

        resize_img = image_from_array.resize((32, 32))

        image_arr_test.append(np.array(resize_img))

        labels_test.append(1)

        

    except AttributeError:

        print("An error occur while reading the image")
train_data = np.array(image_arr_train)

train_labels = np.array(labels_train)

idx = np.arange(train_data.shape[0])

np.random.shuffle(idx)

train_data = train_data[idx]

train_labels = train_labels[idx]
test_data = np.array(image_arr_test)

test_labels = np.array(labels_test)

idxt = np.arange(test_data.shape[0])

np.random.shuffle(idxt)

test_data = test_data[idxt]

test_labels = test_labels[idxt]
print(train_data.shape)

print(train_labels.shape)

print(test_data.shape)

print(test_labels.shape)
test_labels=test_labels.reshape(test_labels.shape[0],1)

train_labels=train_labels.reshape(train_labels.shape[0],1)

print(train_labels.shape)

print(test_labels.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)

number_of_train = X_train.shape[0]

number_of_test = X_test.shape[0]
X_train_flatten = X_train.reshape(number_of_train,X_train.shape[1]*X_train.shape[2]*X_train.shape[3])

X_test_flatten = X_test .reshape(number_of_test,X_test.shape[1]*X_test.shape[2]*X_test.shape[3])

print("X train flatten",X_train_flatten.shape)

print("X test flatten",X_test_flatten.shape)
print("y train: ",Y_train.shape)

print("y test: ",Y_test.shape)
x_train = X_train_flatten

x_test = X_test_flatten

y_train = Y_train

y_test = Y_test

print("x train: ",x_train.shape)

print("x test: ",x_test.shape)

print("y train: ",y_train.shape)

print("y test: ",y_test.shape)
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score

from keras.models import Sequential # initialize neural network library

from keras.layers import Dense # build our layers library

def build_classifier():

    classifier = Sequential() # initialize neural network

    classifier.add(Dense(units = 7, kernel_initializer = 'uniform', activation = 'relu', input_dim = x_train.shape[1]))

    classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu'))

    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    return classifier

classifier = KerasClassifier(build_fn = build_classifier, epochs = 100)

accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 3)

mean = accuracies.mean()

variance = accuracies.std()

print("Accuracy mean: "+ str(mean))

print("Accuracy variance: "+ str(variance))
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)
print("score : ",classifier.score(x_test,y_test))
from sklearn.metrics import confusion_matrix



cm = confusion_matrix(y_test,y_pred)



#%%

f , ax = plt.subplots(figsize=(5,5))



sns.heatmap(cm,annot = True,linewidths =0.5,linecolor ="Red",fmt=".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()