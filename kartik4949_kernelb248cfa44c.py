# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import cv2

import os

train = []

y = []

for i,j,k in os.walk('../input/ph_soil/'):

    for filename in k:

        train.append(cv2.resize(cv2.cvtColor(cv2.imread('../input/ph_soil/'+filename),cv2.COLOR_BGR2RGB),(320,320)))

        y.append(float(filename[:-4]))

        print('../input/ph_soil/'+filename)

        
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(

    horizontal_flip=True)
train  = np.asarray(train,dtype='float32')
train /=255 
y[9]
from sklearn.preprocessing import MinMaxScaler
y  = np.asarray(y)

y.shape
y = np.reshape(y,(40,1))
mm = MinMaxScaler()

modifiedy = mm.fit_transform(y)
from sklearn.externals import joblib

joblib.dump(mm,'minmaxscaler.pkl')
modifiedy = np.reshape(modifiedy,(40))


import numpy as np

datagen.fit(train)
from keras.layers import SeparableConv2D,MaxPooling2D,Dropout,Activation,Dense,Flatten,LeakyReLU,BatchNormalization,Conv2D

from keras.models import Sequential
from keras import backend as K
def custom_activation( x, target_min=5, target_max=10 ) :

    x02 = K.tanh(x) + 1 # x in range(0,2)

    scale = ( target_max-target_min )/2.

    return  x02 * scale + target_min
from keras.optimizers import Adam

from keras import optimizers
def customactivationmodel(optimizer = optimizers.SGD(lr=0.01, nesterov=True)):

    

    model = Sequential()

    model.add(BatchNormalization(input_shape=(320,320,3)))

    model.add(SeparableConv2D(64,(4,4),strides=(1,1)))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))

    model.add(LeakyReLU(alpha=0.3))

    model.add(Dropout(0.2))

    model.add(BatchNormalization())

    model.add(SeparableConv2D(32,(4,4),strides=(1,1)))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))

    model.add(LeakyReLU(alpha=0.3))

    model.add(Dropout(0.2))

    model.add(SeparableConv2D(32,(4,4),strides=(1,1)))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))

    model.add(LeakyReLU(alpha=0.3))

    #model.add(Dropout())

    model.add(SeparableConv2D(16,(4,4),strides=(1,1)))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))

    model.add(LeakyReLU(alpha=0.3))

    model.add(Dropout(0.2))

    model.add(BatchNormalization())

    model.add(SeparableConv2D(8,(4,4),strides=(1,1)))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))

    model.add(LeakyReLU(alpha=0.3))

    #model.add(Dropout())

    model.add(Flatten())

    model.add(Dense(128,activation='relu'))

    model.add(Dense(1,activation='sigmoid'))

    model.compile(optimizer='adam',loss='mean_squared_error')

    return model
def linearactivationmodel():

    

    model = Sequential()

    model.add(BatchNormalization(input_shape=(320,320,3)))

    model.add(Conv2D(64,(4,4),strides=(1,1)))

    

    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))

    model.add(Activation('relu'))

    #model.add(LeakyReLU(alpha=0.3))

    #model.add(Dropout(0.2))

    #model.add(BatchNormalization())

    model.add(Conv2D(32,(4,4),strides=(1,1)))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))

    model.add(Activation('relu'))

    #model.add(LeakyReLU(alpha=0.3))

    model.add(BatchNormalization())

    #model.add(Dropout(0.2))

    model.add(Conv2D(32,(4,4),strides=(1,1)))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))

    model.add(Activation('relu'))

    #model.add(Dropout())

    model.add(Conv2D(16,(4,4),strides=(1,1)))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))

    model.add(Activation('relu'))

    model.add(Dropout(0.2))

    #model.add(BatchNormalization())

    model.add(BatchNormalization())

    model.add(Conv2D(8,(4,4),strides=(1,1)))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))

    #model.add(LeakyReLU(alpha=0.3))

    model.add(Activation('relu'))

    #model.add(Dropout())

    model.add(Flatten())

    model.add(Dense(128,activation='relu'))

    model.add(Dense(1,activation='sigmoid'))

    model.compile(optimizer=Adam(lr=0.0001),loss='binary_crossentropy')

    return model
model = linearactivationmodel()
model.summary()
model.fit_generator(datagen.flow(train,modifiedy,batch_size=1), samples_per_epoch=len(train), epochs=20)

from matplotlib import pyplot as plt

plt.plot()
from sklearn.externals import joblib

joblib.dump(model,'phmodel.sav')
predicted = mm.inverse_transform(model.predict(np.reshape(train,(40,320,320,3))))
y
predicted
import requests 

image_url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSZNkSpHEq-M2B6DcPF2BKvxPgItHyMrKXiRjD8ujAFJNqCMCitpQ"

# URL of the image to be downloaded is defined as image_url 

#r = requests.get('https://mail-attachment.googleusercontent.com/attachment/u/0/?ui=2&ik=aa9cfa7b74&jsver=pd5oUcMKyDc.en.&cbl=gmail_fe_190325.10_p4&view=att&th=%23msg-f:1629585621080920467&disp=zip&permmsgid=msg-f:1629585621080920467&saddbat=ANGjdJ8IEQLz9Aj9fFKEBu-EOJKPRwRmI5JF4Pkj2vw2NCsXPw_WhTwXmP4A42iT4wbDmjfWL5OS7ZU5403GIHTUm8xcf5pV4oYXRY6xoOepmdZNY-6KgSji21sd8A3t3C7Ym9Wy7y8MbsstcS9josCbzTLyhue6f1-l_ec7yfhSl3thyz1f8kXjNU51dyV_OE8JddzqA7Aemaz-CaLuPw2rPMiUVTMNoqLbTI-TvMhtqTzvQMOP0sYTcyWt3LWcUUzh4GQCe9jr8Y_XOjhJuS0a9M9giCKcZX7a_1sZmLv4MxJJ4mjx1phZq7JIHKpGhF4wqFWqmHDc1zUceXK6NDamVONoOuLwa6yNjy96RdPx9wMBRjXnEday18nuBl6RFJrq0D_57KUM4ykg7DkHy2N_6noPMW5djoYMadKwDBpJDEtPUlVt7ZDAbGDAWRzznKUEQQcdEKPb34YgG9Xdothm_WDHVkcWL-OnHa6pwjLQ-TBOqwy7dTWlA0lgv7BEFVUw21CFyvW2_bdb5SqrMv2M_Vice59jcDWkKPihQq_8JDor0KmEimrkIhFEmSHOjKtWbngKYwem6XoUkCCzeZ_ihW39ZGR9o6amN8-cSCrqm5nnOC4KnGjTToga5d4iBPXUFicxmPqoid0KnQ-PXo3aN5sw3vOYHBClk28KIg') # create HTTP response object 

r = requests.get(image_url)

# send a HTTP request to the server and save 

# the HTTP response in a response object called r 

with open("testimage.jpg",'wb') as f: 

  

    # Saving received content as a png file in 

    # binary format 

  

    # write the contents of the response (r.content) 

    # to a new file in binary mode. 

    f.write(r.content)
for i,j,k in os.walk('.'):

    for file in k:

        print(file)
def predict(imgpath):

    test = cv2.resize(cv2.cvtColor(cv2.imread(imgpath),cv2.COLOR_BGR2RGB),(320,320))

    test = test / 255

    return mm.inverse_transform(model.predict(np.reshape(test,(1,320,320,3))))

    
model.predict(np.reshape(train[19],(1,320,320,3)))
mm.inverse_transform(np.reshape(modifiedy[1],(-1,1)))
mm.inverse_transform(model.predict(np.reshape(train[9],(1,320,320,3))))
PATHTOIMAGE = './testimage.jpg'
import matplotlib.pyplot as plt

print('The soil image is :')

plt.imshow(cv2.imread(PATHTOIMAGE)[:,:,[2,1,0]])

print('The PH for the input soil is approx {0:.3f} pH'.format(predict(PATHTOIMAGE)[0][0]))