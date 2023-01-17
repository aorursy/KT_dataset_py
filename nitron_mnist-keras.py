import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



from sklearn.model_selection import StratifiedShuffleSplit

from keras.layers import *

from keras.models import Model



import os

from pathlib import Path

print(os.listdir("../input"))

root = Path("../input")

val_size = 0.2

rseed = 7
# set random seed

np.random.seed(rseed)

from tensorflow import set_random_seed

set_random_seed(rseed)
def normalize(data):

    return np.array(data).reshape(-1,28,28,1)/255



def show_inst(fsize,ims,labels,r=3,c=3):

    fig=plt.figure(figsize=fsize)

    labels=np.array(labels)

    for i in range(len(ims)):

        fig.add_subplot(r,c,i+1)

        plt.imshow(ims[i].reshape(28,28),cmap='gray')

        plt.title(np.argmax(labels[i]))

    plt.tight_layout(True)
train_data = pd.read_csv(root/'train.csv')
test_data = pd.read_csv(root/'test.csv')
train_data.head()
train_data.describe()
train_data.info()
train_data.shape
def num_missing(data):

    print(f"Missing values: {data.isnull().sum().sum()}")
num_missing(train_data)
train_data.label.hist()
test_data.head()
test_data.describe()
test_data.info()
train_enc = pd.get_dummies(train_data,columns=['label'])

train_enc.head()
split = StratifiedShuffleSplit(n_splits=1,test_size=val_size,random_state=rseed)



for train_index,val_index in split.split(train_data,train_data['label']):

    train=train_enc.iloc[train_index]

    val=train_enc.iloc[val_index]

train.shape,val.shape
train_X,train_y=train.loc[:,'pixel0':'pixel783'],train.loc[:,'label_0':'label_9']

val_X,val_y=val.loc[:,'pixel0':'pixel783'],val.loc[:,'label_0':'label_9']
train_X.head()
train_y.head()
train_X=normalize(train_X)

val_X=normalize(val_X)



show_inst((6,6),train_X[:9],train_y[:9])
bs=256
from keras.preprocessing.image import ImageDataGenerator



traingen = ImageDataGenerator(

        rotation_range=20,

        width_shift_range=0.2,

        height_shift_range=0.2,

        shear_range=0.2,

        zoom_range=0.2)
def conv_concat(X, filters):

    X1 = Conv2D(filters,(3,3),padding='same')(X)

    X1 = BatchNormalization(axis=3)(X1)

    X1 = Activation('relu')(X1)

    

    X2 = Conv2D(filters,(5,5),padding='same')(X)

    X2 = BatchNormalization(axis=3)(X2)

    X2 = Activation('relu')(X2)

    

    X3 = Conv2D(filters,(7,7),padding='same')(X)

    X3 = BatchNormalization(axis=3)(X3)

    X3 = Activation('relu')(X3)

    

    X = concatenate([X1,X2,X3],axis=3)

    

    return X



def model(input_shape = (28, 28, 1), classes = 10):

    

    # Define the input as a tensor with shape input_shape

    X_input = Input(input_shape)

    

    X = conv_concat(X_input, 32)

    X = MaxPooling2D(pool_size=(2,2))(X)

    

    X = conv_concat(X, 64)

    X = MaxPooling2D(pool_size=(2,2))(X)

    

    X = conv_concat(X, 128)

    X = MaxPooling2D(pool_size=(4,4))(X)

    

    # dense block

    X = Flatten()(X)

    

    X = Dense(100,activation='relu')(X)

    X = Dropout(rate=0.6)(X)

    

    X = Dense(50,activation='relu')(X)

    X = Dropout(rate=0.4)(X)

    

    X = Dense(10,activation='softmax')(X)

    

    fmodel = Model(inputs=X_input,outputs=X,name='fmodel')



    return fmodel
kmodel = model(train_X.shape[1:])
kmodel.compile('adam','binary_crossentropy',metrics=['accuracy'])
kmodel.fit_generator(traingen.flow(train_X,y=train_y, batch_size=bs),epochs=100,steps_per_epoch=train_X.shape[0]//bs)
val_preds = kmodel.evaluate(val_X,val_y)

print(f'Loss: {val_preds[0]}')

print(f'Val accuracy: {val_preds[1]}')
test = normalize(test_data)

preds = kmodel.predict(test)

labels = np.argmax(preds,axis=1)

sub_df = pd.DataFrame({'ImageId':test_data.index+1,'Label':labels})
show_inst((10,10),test[:25],preds[:25],r=5,c=5)
sub_df.head(10)
sub_df.to_csv('sub.csv',index=False)