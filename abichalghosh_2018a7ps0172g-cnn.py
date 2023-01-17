# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd 

from keras.preprocessing.image import ImageDataGenerator, load_img

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import random

import os

print(os.listdir("/kaggle/input/nnfl-lab-1"))
filenames = os.listdir("/kaggle/input/nnfl-lab-1/training/training")

categories = []

for filename in filenames:

    category = filename.split('_')[0]

    if category == 'chair':

        categories.append(0)

    elif category == 'kitchen':

        categories.append(1)

    elif category == 'knife':

        categories.append(2)

    elif category == 'saucepan':

        categories.append(3)





df = pd.DataFrame({

    'filename': filenames,

    'category': categories

})
df
df['category'].value_counts().plot.bar()
sample = random.choice(filenames)

image = load_img("/kaggle/input/nnfl-lab-1/training/training/"+sample)

plt.imshow(image)
FAST_RUN = False

IMAGE_WIDTH=200

IMAGE_HEIGHT=200

IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)

IMAGE_CHANNELS=3
# from keras.models import Sequential

# from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization



# model = Sequential()



# model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))

# model.add(BatchNormalization())

# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Dropout(0.25))



# model.add(Conv2D(30, (3, 3), activation='relu'))

# model.add(BatchNormalization())

# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Dropout(0.25))



# model.add(Conv2D(30, (3, 3), activation='relu'))

# model.add(BatchNormalization())

# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Dropout(0.25))



# model.add(Flatten())

# model.add(Dense(30, activation='relu'))

# model.add(BatchNormalization())

# model.add(Dropout(0.5))

# model.add(Dense(4, activation='softmax')) # 2 because we have cat and dog classes



# model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])



# model.summary()
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
earlystop = EarlyStopping(patience=10)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=2, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
callbacks = [earlystop, learning_rate_reduction]
df['category'].head()
df["category"] = df["category"].replace({0: 'chair', 1: 'kitchen', 2: 'knife', 3: 'saucepan'}) 
df.head()
train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)

train_df = train_df.reset_index(drop=True)

validate_df = validate_df.reset_index(drop=True)
train_df['category'].value_counts().plot.bar()
validate_df['category'].value_counts().plot.bar()
total_train = train_df.shape[0]

total_validate = validate_df.shape[0]

batch_size=15
print(total_train)

print(total_validate)
import cv2

from sklearn.model_selection import train_test_split
train_images = []       

train_labels = []

shape = (200,200)  

train_path = '/kaggle/input/nnfl-lab-1/training/training/'



for filename in os.listdir('/kaggle/input/nnfl-lab-1/training/training/'):

    if filename.split('.')[1] == 'jpg':

        img = cv2.imread(os.path.join(train_path,filename))

        

        # Spliting file names and storing the labels for image in list

        name=filename.split('_')[0]

        if name=='chair':

            train_labels.append(0)

        elif name=='kitchen':

            train_labels.append(1)

        elif name=='knife':

            train_labels.append(2)

        elif name=='saucepan':

            train_labels.append(3)

        

        # Resize all images to a specific shape

        img = cv2.resize(img,shape)

        

        train_images.append(img)



# Converting labels into One Hot encoded sparse matrix

#train_labels = pd.DataFrame(train_labels).values

train_labels = pd.get_dummies(train_labels).values

# Converting train_images to array

train_images = np.array(train_images)



# Splitting Training data into train and validation dataset

x_train,x_val,y_train,y_val = train_test_split(train_images,train_labels,random_state=1)
train_labels.shape
x_train.shape
x_val.shape
test_images = []

test_labels = []

shape = (200,200)

test_path = '/kaggle/input/nnfl-lab-1/testing/testing'



for filename in os.listdir('/kaggle/input/nnfl-lab-1/testing/testing'):

    if filename.split('.')[1] == 'jpg':

        img = cv2.imread(os.path.join(test_path,filename))

        

        # Spliting file names and storing the labels for image in list

        test_labels.append(filename.split('_')[0])

        

        # Resize all images to a specific shape

        img = cv2.resize(img,shape)

        

        test_images.append(img)

        

# Converting test_images to array

test_images = np.array(test_images)
len(test_labels)
print(train_labels[3])

plt.imshow(train_images[3])
print(train_labels[217])

plt.imshow(train_images[217])
# from keras.models import Sequential

# from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization



# model = Sequential()



# model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))

# model.add(BatchNormalization())

# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Dropout(0.25))



# model.add(Conv2D(64, (3, 3), activation='relu'))

# model.add(BatchNormalization())

# model.add(Conv2D(64, (3, 3), activation='relu'))

# model.add(BatchNormalization())



# model.add(Conv2D(128, (3, 3), activation='relu'))

# model.add(BatchNormalization())

# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Dropout(0.25))



# model.add(Conv2D(128, (3, 3), activation='relu'))

# model.add(BatchNormalization())

# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Dropout(0.25))



# model.add(Conv2D(256, (3, 3), activation='relu'))

# model.add(BatchNormalization())

# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Dropout(0.25))

# model.add(Conv2D(256, (3, 3), activation='relu'))

# model.add(BatchNormalization())

# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Dropout(0.25))



# model.add(Flatten())

# model.add(Dense(512, activation='relu'))

# model.add(BatchNormalization())

# model.add(Dropout(0.5))



# model.add(Dense(128))

# model.add(BatchNormalization())

# model.add(Activation('relu'))

# #Add Dropout

# model.add(Dropout(0.4))



# model.add(Dense(4, activation='softmax')) 



# model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])



# model.summary()
from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization, Input



model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))

model.add(Conv2D(filters=32, kernel_size=(3,3), padding="same", activation="relu"))

#model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

# model.add(Conv2D(filters=32, kernel_size=(3,3), padding="same", activation="relu"))

# #model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))

# model.add(BatchNormalization())

# model.add(MaxPooling2D((2, 2)))

# model.add(Dropout(0.25))



model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))

# model.add(BatchNormalization())

# model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))

#model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.3))



model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))

#model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.4))



model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))

#model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.4))



model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(BatchNormalization())

# model.add(Dense(128, activation='relu'))

# model.add(BatchNormalization())

# model.add(Dense(128, activation='relu'))

# model.add(BatchNormalization())



model.add(Dropout(0.5))



# model.add(Dense(4096, activation="relu"))

# model.add(BatchNormalization())

# model.add(Activation('relu'))

# model.add(Dropout(0.4))



model.add(Dense(4, activation='softmax')) 



model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])



model.summary()
# AlexNet = Sequential()



# #1st Convolutional Layer

# AlexNet.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))

# AlexNet.add(Conv2D(filters=96, input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS), kernel_size=(11,11), strides=(4,4), padding='same'))

# AlexNet.add(BatchNormalization())

# AlexNet.add(Activation('relu'))

# AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))



# #2nd Convolutional Layer

# AlexNet.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1,1), padding='same'))

# AlexNet.add(BatchNormalization())

# AlexNet.add(Activation('relu'))

# AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))



# #3rd Convolutional Layer

# AlexNet.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))

# AlexNet.add(BatchNormalization())

# AlexNet.add(Activation('relu'))



# #4th Convolutional Layer

# AlexNet.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))

# AlexNet.add(BatchNormalization())

# AlexNet.add(Activation('relu'))



# #5th Convolutional Layer

# AlexNet.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))

# AlexNet.add(BatchNormalization())

# AlexNet.add(Activation('relu'))

# AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))



# #Passing it to a Fully Connected layer

# AlexNet.add(Flatten())

# # 1st Fully Connected Layer

# AlexNet.add(Dense(4096, input_shape=(32,32,3,)))

# AlexNet.add(BatchNormalization())

# AlexNet.add(Activation('relu'))

# # Add Dropout to prevent overfitting

# AlexNet.add(Dropout(0.4))



# #2nd Fully Connected Layer

# AlexNet.add(Dense(4096))

# AlexNet.add(BatchNormalization())

# AlexNet.add(Activation('relu'))

# #Add Dropout

# AlexNet.add(Dropout(0.4))



# #3rd Fully Connected Layer

# AlexNet.add(Dense(1000))

# AlexNet.add(BatchNormalization())

# AlexNet.add(Activation('relu'))

# #Add Dropout

# AlexNet.add(Dropout(0.4))



# #Output Layer

# AlexNet.add(Dense(10))

# AlexNet.add(BatchNormalization())

# AlexNet.add(Activation('softmax'))



# AlexNet.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])



# #Model Summary

# AlexNet.summary() 
dataAugmentaion = ImageDataGenerator(rotation_range = 15, zoom_range = 0.20, shear_range = 0.1, horizontal_flip = True, 

width_shift_range = 0.1, height_shift_range = 0.1, rescale=1./255)
# rotation_range=15,

#     rescale=1./255,

#     shear_range=0.1,

#     zoom_range=0.2,

#     horizontal_flip=True,

#     width_shift_range=0.1,

#     height_shift_range=0.1
print(x_train.shape, y_train.shape)
# AlexNet.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# Training the model

history = model.fit(x_train,y_train,epochs=74,batch_size=16,validation_data=(x_val,y_val))

# epochs=3 if FAST_RUN else 10

#history=model.fit_generator(dataAugmentaion.flow(x_train, y_train, batch_size = 32),validation_data = (x_val, y_val), steps_per_epoch = len(x_train) // 32,epochs = 50)
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
evaluate = model.evaluate(x_val,y_val)

print(evaluate)

model.save_weights("checkv10.h5")
predict = model.predict(test_images)
(np.argmax(predict[4]))
outputs=[]

for i in range(len(predict)) :

    temp=[]

    temp.append(test_labels[i])

    temp.append(np.argmax(predict[i]))

    outputs.append(temp)

output=pd.DataFrame(outputs)

#columns=columns={"0": "id", "1": "label"}

#output.rename(str.lower, axis='columns')
output=output.rename(columns={0: "id", 1: "label"})
#output
print(test_labels[42])

print(np.argmax(predict[42]))

plt.imshow(train_images[42])
output.to_csv('sv10final.csv', index=False)
plt.figure(figsize=(20,100))

for n , i in enumerate(list(np.random.randint(0,len(predict),100))) : 

    plt.subplot(20,5,n+1)

    plt.imshow(test_images[i])    

    plt.axis('off')

    classes = {'chair':0 ,'kitchen':1,'knife':2,'saucepan':3}

    def get_img_class(n):

        for x , y in classes.items():

            if n == y :

                return x

    plt.title(get_img_class(np.argmax(predict[i])))
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

ax1.plot(history.history['loss'], color='b', label="Training loss")

ax1.plot(history.history['val_loss'], color='r', label="validation loss")

ax1.set_xticks(np.arange(1, 50, 1))

ax1.set_yticks(np.arange(0, 1, 0.1))



ax2.plot(history.history['accuracy'], color='b', label="Training accuracy")

ax2.plot(history.history['val_accuracy'], color='r',label="Validation accuracy")

ax2.set_xticks(np.arange(1, 50, 1))



legend = plt.legend(loc='best', shadow=True)

plt.tight_layout()

plt.show()
output
# test_filenames = os.listdir("/kaggle/input/nnfl-lab-1/testing/testing")

# test_df = pd.DataFrame({

#     'filename': test_filenames

# })

# nb_samples = test_df.shape[0]
# test_df['category'] = np.argmax(predict, axis=-1)
# sample_test = test_df.head(10)

# sample_test.head()

# plt.figure(figsize=(12, 24))

# for index, row in sample_test.iterrows():

#     filename = row['filename']

#     category = row['category']

#     img = load_img("/kaggle/input/nnfl-lab-1/testing/testing/"+filename, target_size=IMAGE_SIZE)

#     plt.subplot(6, 3, index+1)

#     plt.imshow(img)

#     plt.xlabel(filename + '(' + "{}".format(category) + ')' )

# plt.tight_layout()

# plt.show()
# submission_df = test_df.copy()

# submission_df['id'] = submission_df['filename'].str.split('.').str[0]

# submission_df['label'] = submission_df['category']

# submission_df.drop(['filename', 'category'], axis=1, inplace=True)

# submission_df.to_csv('s1.csv', index=False)
from IPython.display import HTML 

import pandas as pd 

import numpy as np

import base64 

def create_download_link(df, title = "Download CSV file", filename = "data.csv"): 

    csv = df.to_csv(index=False) 

    b64 = base64.b64encode(csv.encode()) 

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(output)