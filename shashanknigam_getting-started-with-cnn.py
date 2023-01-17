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
dftrain = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
def details(df):

    print(df.describe())

    print(df.columns)



details(dftrain)




# def make_directory(x):

#     root = '/kaggle/working'

#     os.mkdir(root+'/'+x)

#     return root+'/'+x

# 

# x = 'training'

# image_dir = make_directory(x)



# for i in range(10):

#     make_directory(x+'/'+str(i))
# converting image from .csv to np.array

z = np.genfromtxt('/kaggle/input/digit-recognizer/train.csv',delimiter=',',skip_header=1)
z.shape # 42000 images of size 28 x 28  with a exta column because of the label of that image at position 0.
import random

x = random.sample(list(z),len(z)) # for randomizing data to remove biasness while splitting data



z = np.array(x) 

z.shape  # checking dimensions after randomizing
# for viewing images(pictorial form)

from matplotlib import pyplot as plt

c =0 

for i in z[:10]:

    c+=1

    l =  i[0]

    y = i[1:]

    y = y.reshape(28,28)

    plt.imshow(y)

    plt.xlabel(l)

    #plt.subplot(2,5,c)

    plt.show()

# saving image in training directory for training and validation directory for validation with a split of 0.2

def image_saving(split_ratio):

    count = 0

    index = int(split_ratio * len(z))

    index = len(z) - index

    for i in z[:index]:

        label = i[0]

        y = i[1:]

        y = y.reshape(28,28)

        plt.imsave('/kaggle/working/training/'+str(label)[0]+'/'+str(count)+'.png',y,format='png')

        count+=1

    print("Total ",count," images saved for training")

    count = 0

    for i in z[index:]:

        label = i[0]

        y = i[1:]

        y = y.reshape(28,28)

        plt.imsave('/kaggle/working/validation/'+str(label)[0]+'/'+str(count)+'.png',y,format='png')

        count+=1

    print('Total ',count,' images saved for validation')



image_saving(0.2)
from keras.models import Sequential

from keras.layers import Conv2D

from keras.layers import MaxPool2D

from keras.layers import Dense

from keras.layers import Flatten

import tensorflow as tf



    
model = Sequential([

    Conv2D(32,(3,3),input_shape=(28,28,3),activation='relu'),

    MaxPool2D(2,2),

    Conv2D(32,(3,3)),

    MaxPool2D(2,2),

    Conv2D(64,(3,3)),

    MaxPool2D(2,2),

    Flatten(),

    Dense(128,activation = tf.nn.relu),

    Dense(10,activation=tf.nn.softmax)

])



model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.summary()
from tensorflow.keras.preprocessing.image import ImageDataGenerator



training_datagen = ImageDataGenerator(rescale=1./255)



training_data = training_datagen.flow_from_directory('/kaggle/working/training',

                                                    target_size=(28,28),

                                                     batch_size = 128,

                                                     classes=['0', '1', '2', '3','4','5','6','7','8','9']

                                                    )



validation_datagen = ImageDataGenerator(rescale=1./255)



validation_data = validation_datagen.flow_from_directory('/kaggle/working/validation',

                                                        target_size=(28,28),

                                                        batch_size=128)
model.fit_generator(training_data,

                   epochs=10,

                    validation_data = validation_data

                   )
plt.plot(model.history.history['accuracy'],label='accuracy')

plt.plot(model.history.history['val_accuracy'],label='validation')

plt.legend()

plt.figure()

plt.plot(model.history.history['loss'],label='loss')

plt.plot(model.history.history['val_loss'],label='val_loss')

plt.legend()

plt.figure()
model.evaluate_generator(validation_data)
dftest = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

dftest.count()
test_z = np.genfromtxt('/kaggle/input/digit-recognizer/test.csv',delimiter=',',skip_header=1)
test_z.shape
# testing on test data to see model correctness change the value in [] for test_z in line 1.

y = test_z[4]

y = y.reshape(28,28)

plt.imshow(y , cmap='gray')

y.shape



from skimage import transform

y_image = transform.resize(y,(28,28,3))

y_image = np.expand_dims(y_image,axis=0)

print(np.argmax(model.predict(y_image)))
res=[]

count=0

for i in test_z:

    x = i[:]

    x = x.reshape(28,28)

    y = transform.resize(x,(28,28,3))

    y = np.expand_dims(y,axis=0)

    res.append(np.argmax(model.predict(y)))

    count+=1

    if count%2000==0:

        print(count," images predicted")

    
# for generating id as per requirement

id = [i for i in range(1,28001)]
# creating dataframe for submission

submission_df = pd.DataFrame({'ImageId':id,'Label':res})
# checking df we just created for submission

submission_df.head()
# Converting dataframe to .csv

submission_df.to_csv('submission1.csv',index=False)