# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

print(os.listdir('../input'))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import glob 

from PIL import Image 

from pathlib import Path





file_dir = Path('../input/asl-rgb-depth-fingerspelling-spelling-it-out/dataset5/*/*')

data_dict = {}

for directory in glob.glob(os.path.abspath(file_dir)):       #os.path.abspath converts the path into byte format 

    images = []

    for files in glob.glob(directory+'/color_*.png'):

        images.append(files)

    data_dict.setdefault(directory[-1], images)
import matplotlib.pyplot as plt

import cv2





f, ax = plt.subplots(4,6, figsize = (20,30))



for i , (k, v) in enumerate(data_dict.items()):

    img = cv2.imread(v[0])

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (64,64))

    ax[i//6, i%6].imshow(img)

    ax[i//6, i%6].axis('off')



plt.show()
images = []

labels = []

for k,v in data_dict.items():

        for file in v:

            images.append(file)

            labels.append(k)
data_list = list(zip(images,labels))

data_df = pd.DataFrame(data_list, columns = ['Image', 'Label'])

data_df
from sklearn.model_selection import train_test_split

from sklearn.utils import shuffle

from sklearn.preprocessing import LabelBinarizer



data = data_df.drop(columns = 'Label')





labels = data_df['Label']

lb = LabelBinarizer()

encoded_labels = lb.fit_transform(labels)





#shuffeling the data 

data, encoded_labels = shuffle(data, encoded_labels, random_state = 0)



X_train, X_test, y_train, y_test = train_test_split(data, encoded_labels, test_size = 0.25, random_state = 20)

print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)
y_train[0]
X_train['Image'][1:10]
X_train = np.array(X_train)

y_train = np.array(y_train)

X_test = np.array(X_test)

y_test = np.array(y_test)










def data_generator(data,encoded_labels, batch_size):

    num_samples = len(data)

    while True:



        for offset in range(0, num_samples, batch_size):

            batch_samples = data[offset: offset+batch_size]

            label_samples = encoded_labels[offset: offset+batch_size]

            X = []

            y = []

            for batch_sample in batch_samples:

                img_name = batch_sample[0]



                img = cv2.imread(str(img_name))

                img = cv2.resize(img, (128,128))

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                img = img.astype(np.float32)/255

                

                X.append(img)

            for label in label_samples:

                y.append(label)

                

                

                

            X = np.array(X)

            y = np.array(y)

            

            yield X, y
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dense, Flatten, Dropout, Activation

from keras.models import Sequential

model = Sequential()

model.add(Conv2D(32, (3,3), input_shape = (128, 128, 3), activation = 'relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(2,2))

model.add(Dropout(0.25))



model.add(Conv2D(64, (3,3), activation = 'relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(2,2))

model.add(Dropout(0.25))



model.add(Conv2D(128, (3,3), activation = 'relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(2,2))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(512, activation = 'relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(Dense(24, activation = 'softmax'))



model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])

model.summary()
bactsize = 40

train_data_gen = data_generator(data = X_train,encoded_labels = y_train, batch_size = bactsize)

val_data_gen = data_generator(data = X_test, encoded_labels = y_test, batch_size = bactsize)
X_any, y_any = next(train_data_gen)

len(X_any)

len(y_any)

f, ax = plt.subplots(10,4, figsize = (40,60))

new_labels = []



for rows in y_any:

    row_max = np.argmax(rows)

    new_labels.append(row_max)







for i in range(40):

    

    img = X_any[i]    

    ax[i//4, i%4].imshow(img)

    ax[i//4, i%4].set_title(new_labels[i], fontsize = 30)

    

    ax[i//4, i%4].set_aspect('auto')

    ax[i//4, i%4].axis('off')

plt.show()

        



    

    

    
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

earlystopping = EarlyStopping(patience = 10)

learning_rate_reduction = ReduceLROnPlateau(moniter = 'val_acc', patience = 10, verbose = 1, factor = 0.5,

                                           min_lr = 0.0001)

mycallbacks = [earlystopping, learning_rate_reduction]
total_train = X_train.shape[0]

total_val = X_test.shape[0]

validation_steps=total_val//bactsize

steps_per_epoch=total_train//bactsize

callbacks=mycallbacks
eph = 10

history = model.fit_generator(

    train_data_gen, 

    epochs=eph,

    validation_data=val_data_gen,

    validation_steps=total_val//bactsize,

    steps_per_epoch=total_train//bactsize,

    callbacks=mycallbacks

)
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model Accuracy')

plt.ylabel('Epochs')

plt.xlabel('Accuracy')

plt.legend(['Train, Validation'], loc = 'best')

plt.show()

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['Trian', 'Validation'], loc='upper left')

plt.show()
custom_img_dir = Path('../input/testimage/')

img_dir = os.listdir(custom_img_dir)

img = cv2.imread(os.path.join(custom_img_dir, img_dir[0]))

img = cv2.resize(img, (128,128))

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img = img.astype(np.float32)/255

custom_img = np.array(img)

plt.imshow(custom_img)

custom_img.shape
img_list = []

img_list.append(custom_img)

img_list = np.array(img_list)

img_list.shape
pred = model.predict_proba(img_list)

pred = np.argmax(pred)

pred