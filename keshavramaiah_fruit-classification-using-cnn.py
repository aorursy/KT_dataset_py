import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from keras.layers import Activation,Dropout,Dense,Conv2D,AveragePooling2D,Flatten,ZeroPadding2D,MaxPooling2D

from keras import optimizers

from sklearn.model_selection import train_test_split

from keras.models import Sequential

import seaborn as sns

from sklearn.metrics import accuracy_score

from keras.utils.np_utils import to_categorical

from keras.preprocessing.image import img_to_array

from keras.preprocessing.image import load_img

import math

import cv2

import imageio

from os import listdir

import warnings

import filecmp

from PIL import Image
def rotate(image, angle):

  image_center = tuple(np.array(image.shape[1::-1]) / 2)

  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)

  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

  return result
apple=[]

prom=[]

orange=[]

IMG_SHAPE = 50

fruit_images=[]

fruit_labels=[]

#listdir("../input/train/Train")

base_path = "../input/train/Train/"

promogranate=base_path+'pomegranate/'

for file in listdir(promogranate):

    file_path = promogranate + file

    if file.endswith('.jpg'):

            image = imageio.imread(file_path)

            if len(image.shape) > 2 and image.shape[2] == 4:

                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

            image=cv2.resize(image,(50,50))

            image=image/255

            rotated45 = rotate(image,45)

            rotated75 =rotate(image,75)

            rotated120=rotate(image,120)

           # print(image.shape)

            prom.append(image)

            prom.append(rotated45)

            prom.append(rotated75)

            prom.append(rotated120)

            fruit_images.append(image)

            fruit_images.append(rotated45)

            fruit_images.append(rotated75)

            fruit_images.append(rotated120)

            fruit_labels.append(0)

            fruit_labels.append(0)

            fruit_labels.append(0)

            fruit_labels.append(0)

app=base_path+'apples/'

for file in listdir(app):

    if file.endswith('.jpg'):

        file_path = app + file

        image = imageio.imread(file_path)

        if len(image.shape) > 2 and image.shape[2] == 4:

                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        image=cv2.resize(image,(50,50))

        image=image/255

       # print(image.shape)

        rotated45 = rotate(image,45)

        rotated75 =rotate(image,75)

        rotated120=rotate(image,120)

           # print(image.shape)

        apple.append(image)

        apple.append(rotated45)

        apple.append(rotated75)

        apple.append(rotated120)

        fruit_images.append(image)

        fruit_images.append(rotated45)

        fruit_images.append(rotated75)

        fruit_images.append(rotated120)

        fruit_labels.append(2)

        fruit_labels.append(2)

        fruit_labels.append(2)

        fruit_labels.append(2)

oranj=base_path+'oranges/'

for file in listdir(oranj):

    if file.endswith('.jpg'):

        file_path = oranj + file

        image = imageio.imread(file_path)

        if len(image.shape) > 2 and image.shape[2] == 4:

                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        image=cv2.resize(image,(50,50))

        image=image/255

        rotated45 = rotate(image,45)

        rotated75 =rotate(image,75)

        rotated120=rotate(image,120)

           # print(image.shape)

        orange.append(image)

        orange.append(rotated45)

        orange.append(rotated75)

        orange.append(rotated120)

        fruit_images.append(image)

        fruit_images.append(rotated45)

        fruit_images.append(rotated75)

        fruit_images.append(rotated120)

        fruit_labels.append(1)

        fruit_labels.append(1)

        fruit_labels.append(1)

        fruit_labels.append(1)
def ShowFirstFive(images_arr,title):



    fig, axes = plt.subplots(1, 5, figsize=(20,20))



    for img, ax in zip( images_arr, axes):

        ax.imshow(img)

        ax.set_title(title,fontsize=20)

    plt.tight_layout()

    plt.show()

ShowFirstFive(apple,'apple')

ShowFirstFive(prom,'promogranete')

ShowFirstFive(orange,'orange')

def reorder(old_list,order):

    new_list = []

    for i in order:

        new_list.append(old_list[i])

    return new_list



np.random.seed(seed=42)

indices = np.arange(len(fruit_labels))

np.random.shuffle(indices)

indices = indices.tolist()

fruit_labels = reorder(fruit_labels,indices)

fruit_images = reorder(fruit_images,indices)

image_array = np.array(fruit_images)

label_array = np.array(fruit_labels)
label_array=to_categorical(label_array,3)
X_train, X_test, y_train, y_test = train_test_split(image_array,label_array, test_size=0.2)

X_train, X_val, y_train, y_val=train_test_split(X_train, y_train, test_size=0.3)
model =  Sequential([

    

    #convolutional layers

    Conv2D(32, (3,3), activation='relu', input_shape=(50,50,3),padding='same'),

    MaxPooling2D(2, 2),

    Conv2D(64, (3,3), activation='relu',padding='same'),

    MaxPooling2D(2,2),  

    Conv2D(128, (3,3), activation='relu',padding='same'),

    MaxPooling2D(2,2),

    

    # dense layer

    Flatten(),

    Dropout(0.50),

    Dense(128, activation='relu'),

    Dense(3, activation='softmax')

])



l=[25]

for i in range(len(l)):

    model =  Sequential([

    

    #convolutional layers

    Conv2D(32, (3,3), activation='relu', input_shape=(50,50,3),padding='same'),

    MaxPooling2D(2, 2),

    Conv2D(64, (3,3), activation='relu',padding='same'),

    MaxPooling2D(2,2),  

    Conv2D(128, (3,3), activation='relu',padding='same'),

    MaxPooling2D(2,2),

    

    # dense layer

    Flatten(),

    Dropout(0.50),

    Dense(128, activation='relu'),

    Dense(3, activation='softmax')

    ])

    model.compile(optimizer='adam', 

              loss='categorical_crossentropy',

              metrics=['accuracy'])

    model.fit(X_train,y_train,epochs=l[i],verbose=1)

    modelsub=model.predict(X_val,batch_size=None, verbose=1)

    c=0

    f=0

    for i in range(modelsub.shape[0]):

        if np.argmax(modelsub[i])==np.argmax(y_val[i]):

            c+=1

        else:

            f+=1

    accuracy=c/(c+f)

    print('Test Accuracy:',accuracy*100)

   
modelsub=model.predict(X_test,batch_size=None, verbose=1)

c=0

f=0

for i in range(modelsub.shape[0]):

    if np.argmax(modelsub[i])==np.argmax(y_test[i]):

        c+=1

    else:

        f+=1

    accuracy=c/(c+f)

print('Test Accuracy:',accuracy*100)
def get_images(ids, filepath):

    arr = []

    for i in range(len(ids)):

        image = plt.imread(filepath + ids[i]+'.jpg')

        if len(image.shape) > 2 and image.shape[2] == 4:

            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        image=cv2.resize(image,(50,50))

        image=image/255

        arr.append(image)

    

    arr = np.array(arr)

    return arr
sub_df=pd.read_csv('../input/sample_submission_q2.csv')

test_id = sub_df['ID']

test_id=test_id.astype(str)

test_path='../input/test/Test/'

test = get_images(ids=test_id, filepath=test_path)

l=[]

iid=np.arange(1,113,1)



c=model.predict(test,batch_size=None, verbose=1)

for i in range(len(c)):

    l.append(np.argmax(c[i]))

df=pd.DataFrame({'ID':iid,'Category':l

},index=iid)

print(l)

df.to_csv("q2.csv",index=False)