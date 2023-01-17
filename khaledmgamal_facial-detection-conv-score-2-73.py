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


train_data = pd.read_csv('/kaggle/input/facial-keypoints-detection/training/training.csv')  

test_data = pd.read_csv('/kaggle/input/facial-keypoints-detection/test/test.csv')

lookid_data = pd.read_csv('/kaggle/input/facial-keypoints-detection/IdLookupTable.csv')
train_data.head().T
import matplotlib.pyplot as plt



def show_img(image,loc,y_min,y_max):

    plt.imshow(image.reshape(96,96),cmap='gray')

    plt.scatter((loc[0::2]*y_max)+y_min, (loc[1::2]*y_max)+y_min , marker='x', s=10)

    plt.show()



    
def data_preprocess(train_data,is_test):

    train_data.isnull().any().value_counts()

    train_data=train_data.fillna(method = 'ffill')

    train_data.isnull().any().value_counts()#removing None

    imgs=[]

    for i in range(len(train_data)):#preparing X 

       img=train_data['Image'][i].split(' ')

       img=[0 if x=='' else int(x) for x in img ]

       imgs.append(img)

    imgs=np.array(imgs,dtype = 'float')

    images=imgs.reshape([-1,96,96,1])

    X_train=images/255

    if is_test==True:

        return X_train

    else:

       training=train_data.drop('Image',axis=1)#prepearing y

       y_train=[]

       for i in range(0,len(train_data)):

          y=training.iloc[i,:]

          y_train.append(y)

       y_train=np.array(y_train,dtype = 'float')

       y_min=y_train.min()

       y_max=y_train.max()

       y_train=(y_train-y_train.min())/y_train.max()

       #print(y_train.min(),y_train.max())

       return X_train,y_train ,y_min,y_max

    

X_train,y_train,y_min,y_max=data_preprocess(train_data,False)

show_img(X_train[1],y_train[1],y_min,y_max)
from keras.models import Sequential

from keras.layers import Dense , Dropout , Lambda, Flatten,Conv2D,MaxPooling2D,Activation,BatchNormalization,LeakyReLU

from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import Adam ,RMSprop

from keras.models import load_model



model=Sequential()

model.add(Conv2D(filters=16, kernel_size=2, strides=(1, 1), padding='same',input_shape=(96, 96,1)))

model.add(Dropout(0.1))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))

#model.add(Activation('relu'))

model.add(Conv2D(filters=32, kernel_size=2, strides=(1, 1), padding='same'))

model.add(Dropout(0.1))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))

#model.add(Activation('relu'))

model.add(Conv2D(filters=64, kernel_size=2, strides=(1, 1), padding='same'))

model.add(Dropout(0.2))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))

#model.add(Activation('relu'))

model.add(Conv2D(filters=128, kernel_size=2, strides=(1, 1), padding='same'))

model.add(LeakyReLU(alpha = 0.2))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))

model.add(Conv2D(filters=256, kernel_size=2, strides=(1, 1), padding='same'))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))

model.add(Flatten())

model.add(Dense(512))

model.add(Dropout(0.2))

#model.add(LeakyReLU(alpha = 0.1))

model.add(Activation('relu'))

model.add(Dense(128))

#model.add(LeakyReLU(alpha = 0.1))

model.add(Activation('relu'))

model.add(Dense(30))



model.compile(optimizer='adam',

              loss='mse',

              metrics=['accuracy'])

model.summary()
History=model.fit(X_train, y_train, epochs=100,batch_size=128, validation_split=0.2)
loss = History.history['loss']

val_loss = History.history['val_loss']



        

plt.plot(loss,linewidth=1,label="train:")

plt.legend()

plt.grid()

plt.yscale("log")

plt.xlabel("epoch")

plt.ylabel("log loss")

plt.show()

plt.plot(val_loss,linewidth=1,label="val:")





plt.legend()

plt.grid()

plt.yscale("log")

plt.xlabel("epoch")

plt.ylabel("log loss")

plt.show()
X_test=data_preprocess(test_data,is_test=True)

y_pred = model.predict(X_test)
show_img(X_test[1],y_pred[1],y_min,y_max)


landmark_dict={}

lookid_list = list(lookid_data['FeatureName'])



for f in list(lookid_data['FeatureName']):

    landmark_dict.update({f:lookid_list.index(f)})
ImageId = lookid_data["ImageId"]

FeatureName = lookid_data["FeatureName"]

RowId = lookid_data["RowId"]

y_pred=(y_pred*y_max)+y_min

for i in range(y_pred.shape[0]):

    for j in range(y_pred.shape[1]):

         if y_pred[i][j]>96 :

            y_pred[i][j]=96        

submit = []

for rowId,irow,landmark in zip(RowId,ImageId,FeatureName):

    submit.append([rowId,y_pred[irow-1][landmark_dict[landmark]]])

    

submit = pd.DataFrame(submit,columns=["RowId","Location"])

    ## adjust the scale 

print(submit.shape)



submit.to_csv("submision13.csv",index=False) 
from IPython.display import FileLink

FileLink(r'submision13.csv')