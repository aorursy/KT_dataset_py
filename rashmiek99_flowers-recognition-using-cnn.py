# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



import cv2    

from tqdm import tqdm

import random as random

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from keras.utils import to_categorical

from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPool2D,BatchNormalization

from keras.models import Sequential

from keras.losses import categorical_crossentropy

from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator



import warnings

warnings.filterwarnings('always')

warnings.filterwarnings('ignore')



# Any results you write to the current directory are saved as output.
daisy_dir='../input/flowers-recognition/flowers/flowers/daisy'

sunflower_dir='../input/flowers-recognition/flowers/flowers/sunflower'

tulip_dir='../input/flowers-recognition/flowers/flowers/tulip'

dandi_dir='../input/flowers-recognition/flowers/flowers/dandelion'

rose_dir ='../input/flowers-recognition/flowers/flowers/rose'
IMG_SIZE= 150

predicters = []

target = []
def read_image(label,DIR):

    

    for img in tqdm(os.listdir(DIR)):

    #for img in DIR:

        path = os.path.join(DIR,img)

        

        _, ftype = os.path.splitext(path)

        if ftype == ".jpg":

       

            image = cv2.imread(path,cv2.IMREAD_COLOR)

            image = cv2.resize(image, (IMG_SIZE,IMG_SIZE),interpolation = cv2.INTER_AREA)

            predicters.append(np.array(image))

            target.append(str(label))



read_image("Daisy",daisy_dir)

read_image("Sunflower",sunflower_dir)

read_image("Tulip",tulip_dir)

read_image("Dandelion",dandi_dir)

read_image("Rose",rose_dir)

len(predicters),len(target)

#Display few images



fig,ax = plt.subplots(5,3)

fig.set_size_inches(12,12)



for i in range(5):

    for j in range(3):

        l = random.randint(0,len(target))

        ax[i,j].imshow(predicters[l])

        ax[i,j].set_title(target[l])

        

plt.tight_layout()
encoder = LabelEncoder()



X = np.array(predicters)

X = X/255



y = encoder.fit_transform(target)

y = to_categorical(y,5)



print(X.shape)

print(y.shape)
X_train,X_valid,y_train,y_valid = train_test_split(X,y,test_size=0.2,random_state=42)

X_train,X_test,y_train,y_test = train_test_split(X_train,y_train,test_size=0.2,random_state=42)



X_train.shape,y_train.shape,X_valid.shape,y_valid.shape,X_test.shape,y_test.shape
batch_size = 64

epochs = 20

num_classes = y.shape[1]
model = Sequential()



model.add(Conv2D(64, kernel_size=(3, 3),activation='relu',padding='same',input_shape=(IMG_SIZE,IMG_SIZE,3)))

model.add(MaxPool2D((2, 2),padding='same'))

model.add(BatchNormalization())

model.add(Dropout(0.2))



model.add(Conv2D(128,kernel_size= (3, 3), activation='relu',padding='same'))

model.add(MaxPool2D(pool_size=(2, 2),padding='same'))

model.add(BatchNormalization())

model.add(Dropout(0.2))



model.add(Conv2D(128, kernel_size=(3, 3), activation='relu',padding='same'))

model.add(MaxPool2D(pool_size=(2, 2),padding='same'))

model.add(BatchNormalization())

model.add(Dropout(0.3))



model.add(Conv2D(256, kernel_size=(3, 3), activation='relu',padding='same'))

model.add(MaxPool2D(pool_size=(2, 2),padding='same'))

model.add(BatchNormalization())

model.add(Dropout(0.4))



model.add(Conv2D(512, kernel_size=(3, 3), activation='relu',padding='same'))

model.add(MaxPool2D(pool_size=(2, 2),padding='same'))

model.add(BatchNormalization())

model.add(Dropout(0.4))



model.add(Flatten())

model.add(Dense(1024, activation='relu'))

model.add(Dropout(0.3))

model.add(BatchNormalization())



model.add(Dense(num_classes, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'])
imagegen = ImageDataGenerator(featurewise_center=False,

                             samplewise_center=False,

                             featurewise_std_normalization=False,

                             samplewise_std_normalization=False,

                            rotation_range=60,

                              zoom_range=0.1,

                              width_shift_range=0.1,

                              height_shift_range=0.1,

                              shear_range=0.1,

                              fill_mode='reflect')

imagegen.fit(X_train)

                              
model_dropout = model.fit_generator(imagegen.flow(X_train,y_train, batch_size=batch_size),epochs=epochs,verbose=1,

                          validation_data=(X_valid, y_valid),steps_per_epoch=X_train.shape[0] // batch_size

                          )
test_eval = model.evaluate(X_test, y_test, verbose=1)
print("Loss=",test_eval[0])

print("Accuracy=",test_eval[1])
y_pred = model.predict(X_test)

y_pred = np.argmax(y_pred,axis=1).astype(int)



y_true = np.argmax(y_test,axis = 1).astype(int)
corr=[]

incorr=[]

corr_count=0

incorr_count=0



for i in range(len(y_test)):

    if(y_pred[i]==y_true[i]):

        corr.append(i)

        corr_count+=1

    else:

        incorr.append(i)

        incorr_count+=1

        

print("Found %d correct flowers" %(corr_count))

print("Found %d incorrect flowers" %(incorr_count))
fig,ax = plt.subplots(num_classes,4)

fig.set_size_inches(15,15)



count = 0

for i in range (num_classes):

    for j in range (4):

        

        ax[i,j].imshow(X_test[corr[count]])

        

        ax[i,j].set_title("Actual Flower : "+str(encoder.inverse_transform([y_true[corr[count]]])) +  "\n" + "Predicted Flower : "+str(encoder.inverse_transform([y_pred[corr[count]]])))

        

        count+=1

        

plt.tight_layout()  
fig,ax = plt.subplots(num_classes,4)

fig.set_size_inches(15,15)

count = 0

for i in range (num_classes):

    for j in range (4):

        ax[i,j].imshow(X_test[incorr[count]])        

        ax[i,j].set_title("Actual Flower : "+str(encoder.inverse_transform([y_true[incorr[count]]])) +  "\n" + "Predicted Flower : "+str(encoder.inverse_transform([y_pred[incorr[count]]])))



        count+=1

        

plt.tight_layout()