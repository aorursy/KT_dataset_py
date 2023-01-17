

#Libraries for data manipulation

import numpy as np #Linear Algebra

import pandas as pd #Data Processing



#for cnn

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.optimizers import Adam

from keras.layers.normalization import BatchNormalization

from keras.utils import np_utils

from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D

from keras.layers.advanced_activations import LeakyReLU 

from keras.preprocessing.image import ImageDataGenerator



#for data visulization

from matplotlib import pyplot as plt

%matplotlib inline



train_data=pd.read_csv('../input/training/training.csv')

test_data=pd.read_csv("../input/test/test.csv")

lookid_data = pd.read_csv("../input/IdLookupTable.csv")
#Number of rows and columns in the training data

train_data.shape
#Name of the columns in the training data

train_data.columns
#Number of rows and columns in the test data

test_data.shape
#Name of the columns in the test data

test_data.columns
train_data.head().T
#checking for missing values

train_data.isnull().sum()
train_data.fillna(method='ffill',inplace=True)


#Converting image into an 1D list of pixels

images=[]

for i in range(train_data.shape[0]):

    image=['0'if x=='' else x for x in train_data['Image'][i].split(' ')]

    images.append(image)



test_images=[]

for i in test_data['Image']:

    test_images.append(['0' if x==' ' else x for x in i.split(' ') ])



train_data.drop(['Image'],axis=1,inplace=True)    
#convert 1d array of images to 2d array

images=np.array(images,dtype=float)

images=images.reshape(-1,96,96)

test_images=np.array(test_images,dtype=float)

test_images=test_images.reshape(-1,96,96)
print(images[0])


#Let's visualize some columns

f,((ax1,ax2),(ax3,ax4))=plt.subplots(2,2,sharex='col',sharey='row')

ax1.imshow(images[0])

ax2.imshow(images[1])

ax3.imshow(images[2])

ax4.imshow(images[3])

#plt.show()
#for our cnn,we need to convert 2d array into 3d array.

images=images.reshape(-1,96,96,1)
model=Sequential()

model.add(Conv2D(32,(3,3),input_shape=(96,96,1)))

model.add(BatchNormalization(axis=-1))

model.add(Activation('relu'))

model.add(Conv2D(32,(3,3)))

model.add(BatchNormalization(axis=-1))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))

model.add(BatchNormalization(axis=-1))

model.add(Activation('relu'))

model.add(Conv2D(64,(3,3)))

model.add(BatchNormalization(axis=-1))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(512))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Dropout(0.2))

model.add(Dense(30))

#compile the model

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(images,train_data,epochs=1000,batch_size=128,validation_split=0.2)
#some images of test data.

from matplotlib import pyplot as plt

%matplotlib inline

f,((ax1,ax2),(ax3,ax4))=plt.subplots(2,2,sharex='col',sharey='row')

ax1.imshow(test_images[0])

ax2.imshow(test_images[1])

ax3.imshow(test_images[2])

ax4.imshow(test_images[3])
test_images=test_images.reshape(-1,96,96,1)

result=model.predict(test_images)
lookup=pd.read_csv('../input/IdLookupTable.csv')

lookup.head(5)
#code for preparation of submission file

a=list(train_data.columns)



indexes=[]

for i in lookup['FeatureName']:

    indexes.append(a.index(i))

imageid=[]

for i in lookup['ImageId']:

        imageid.append(i-1)

answer=[]

for image,feature in zip(imageid,indexes):

    answer.append(result[image][feature])



rowid=range(1,len(answer)+1)

answer=pd.Series(answer,name='RowId')

rowid=pd.Series(rowid,name='Location')

submission=pd.concat([rowid,answer],axis=1)

submission.to_csv('final.csv',index=False)
