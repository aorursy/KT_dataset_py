import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from IPython.display import clear_output

from time import sleep

import os
Train_Dir = '../input/training.zip'

Test_Dir = '../input/test.zip'

lookid_dir = '../input/IdLookupTable.csv'

train_data = pd.read_csv(Train_Dir)  

test_data = pd.read_csv(Test_Dir)

lookid_data = pd.read_csv(lookid_dir)

os.listdir('../input')
print("shapes","_"*50)

print("test:",test_data.shape)

print("train:",train_data.shape)

print("lookup:",lookid_data.shape)

print("columns","_"*50)

print("test:",test_data.columns)

print("train:",train_data.columns)

print("lookup:",lookid_data.columns)
test_data.head(3)
train_data.head().T
lookid_data.head(10)
train_data.isnull().any().value_counts()


train_data.fillna(method = 'ffill',inplace = True)

#train_data.reset_index(drop = True,inplace = True)

train_data.isnull().any().value_counts()


imag = []

for i in range(0,7049):

    img = train_data['Image'][i].split(' ')

    img = ['0' if x == '' else x for x in img]

    imag.append(img)

    

    
image_list = np.array(imag,dtype = 'float')

X_train = image_list.reshape(-1,96,96,1)



plt.imshow(X_train[0].reshape(96,96),cmap='gray')

plt.show()
training = train_data.drop('Image',axis = 1)



y_train = []

for i in range(0,7049):

    y = training.iloc[i,:]



    y_train.append(y)

y_train = np.array(y_train,dtype = 'float')



# from keras.layers import Conv2D,Dropout,Dense,Flatten

# from keras.models import Sequential



# model = Sequential([Flatten(input_shape=(96,96)),

#                          Dense(128, activation="relu"),

#                          Dropout(0.1),

#                          Dense(64, activation="relu"),

#                          Dense(30)

#                          ])

from keras.layers.advanced_activations import LeakyReLU

from keras.models import Sequential, Model

from keras.layers import Activation, Convolution2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout, Conv2D,MaxPool2D, ZeroPadding2D
model = Sequential()



# layer set 1

model.add(Convolution2D(32, (3,3), padding='same', use_bias=False, input_shape=(96,96,1)))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())



model.add(Convolution2D(32, (3,3), padding='same', use_bias=False))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2)))

# result 1 image is converted into 48 X 48 X 32 = 73K 



# layer set 2

model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())



model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2)))

# outputs 24 X 24 X 64 = 36K



# layer set 3

model.add(Convolution2D(96, (3,3), padding='same', use_bias=False))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())



model.add(Convolution2D(96, (3,3), padding='same', use_bias=False))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2)))

# outputs 12 X 12 X 96 = 14K



#layer set 4

model.add(Convolution2D(128, (3,3),padding='same', use_bias=False))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())



model.add(Convolution2D(128, (3,3),padding='same', use_bias=False))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2)))

# outputs 6 X 6 X 128 = 4K



#layer set 5

model.add(Convolution2D(256, (3,3),padding='same',use_bias=False))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())



model.add(Convolution2D(256, (3,3),padding='same',use_bias=False))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2)))

# outputs 3 X 3 X 256 = 2K



# different set 

model.add(Convolution2D(512, (3,3), padding='same', use_bias=False))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())



model.add(Convolution2D(512, (3,3), padding='same', use_bias=False))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())

# outputs 3 X 3 X 512 = 4K



# Normal layer

model.add(Flatten())

model.add(Dense(512,activation='relu'))

model.add(Dropout(0.1))

model.add(Dense(30))

model.summary()
model.compile(optimizer='adam', 

              loss='mean_squared_error',

              metrics=['mae'])

# mean absolute error
model.fit(X_train,y_train,epochs = 50,batch_size = 256,validation_split = 0.2)
model.save('saved_model/model')
# import tensorflow as tf

# model = tf.keras.models.load_model('saved_model/model')
#preparing test data

timag = []

for i in range(0,1783):

    timg = test_data['Image'][i].split(' ')

    timg = ['0' if x == '' else x for x in timg]

    

    timag.append(timg)
timage_list = np.array(timag,dtype = 'float')

X_test = timage_list.reshape(-1,96,96,1) 
plt.imshow(X_test[0].reshape(96,96),cmap = 'gray')

plt.show()
pred = model.predict(X_test)
pred[0]
lookid_list = list(lookid_data['FeatureName'])

imageID = list(lookid_data['ImageId']-1)

pre_list = list(pred)
rowid = lookid_data['RowId']

rowid=list(rowid)
feature = []

for f in list(lookid_data['FeatureName']):

    feature.append(lookid_list.index(f))
preded = []

for x,y in zip(imageID,feature):

    preded.append(pre_list[x][y])
rowid = pd.Series(rowid,name = 'RowId')
loc = pd.Series(preded,name = 'Location')
submission = pd.concat([rowid,loc],axis = 1)

submission.to_csv('face_key_detection_submission.csv',index = False)