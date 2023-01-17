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
import os

print(os.listdir(os.getcwd()))
#Ad majorem dei gloriam



from keras.preprocessing.image import ImageDataGenerator

from tensorflow import keras

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Dropout,BatchNormalization, Input, Activation, GlobalAveragePooling2D

from keras.optimizers import Adam

from keras.models import Model

from keras.callbacks import ReduceLROnPlateau,TensorBoard,ModelCheckpoint,EarlyStopping 

import sys

import time

from keras.applications import densenet

import os

#import numpy as np

import pandas as pd

import cv2

import matplotlib.pyplot as plt

import gc





###--- FUNCTIONS

class MultiOutputDataGenerator(keras.preprocessing.image.ImageDataGenerator):

    

    def flow(self,x,y=None,batch_size=32,shuffle=True,sample_weight=None,seed=None,save_to_dir=None,save_prefix='',save_format='png',subset=None):



        targets = None

        target_lengths = {}

        ordered_outputs = []

        for output, target in y.items():

            if targets is None:

                targets = target

            else:

                targets = np.concatenate((targets, target), axis=1)

            target_lengths[output] = target.shape[1]

            ordered_outputs.append(output)





        for flowx, flowy in super().flow(x, targets, batch_size=batch_size,

                                         shuffle=shuffle):

            target_dict = {}

            i = 0

            for output in ordered_outputs:

                target_length = target_lengths[output]

                target_dict[output] = flowy[:, i: i + target_length]

                i += target_length



            yield flowx, target_dict



def SimpleNet(inp):

    

    #Model 1

    #L1

    x=Conv2D(32, (3,3),activation="relu", padding='SAME')(inp)

    x=Conv2D(32, (3,3),activation="relu", padding="SAME")(x)

    x=Conv2D(32, (3,3),activation="relu", padding="SAME")(x)

    x=MaxPool2D(pool_size=(2,2))(x)



    #L2

    x=Conv2D(64,(3,3),activation="relu", padding="SAME")(x)

    x=Conv2D(64,(3,3),activation="relu", padding="SAME")(x)

    x=Conv2D(64,(3,3),activation="relu", padding="SAME")(x)

    x=MaxPool2D(pool_size=(2,2))(x)

    x=Dropout(0.10)(x)



    #L3

    x=Conv2D(128,(3,3),padding="SAME",activation="relu")(x)

    x=Conv2D(128,(3,3),padding="SAME",activation="relu")(x)

    x=Conv2D(128,(3,3),padding="SAME",activation="relu")(x)

    x=MaxPool2D(pool_size=(2,2))(x)

    x=Dropout(0.20)(x)

    

    #L4

    x=Conv2D(256,(3,3),padding="SAME",activation="relu")(x)

    x=Conv2D(256,(3,3),padding="SAME",activation="relu")(x)

    x=Conv2D(256,(3,3),padding="SAME",activation="relu")(x)

    x=MaxPool2D(pool_size=(2,2))(x)

    x=Dropout(0.20)(x)

    



    #flatten

    x=Flatten()(x)

    x=Dense(1152, activation = "relu")(x)

    x=Dropout(rate=0.25)(x)

    dense=Dense(324, activation = "relu")(x)

    dense=Dropout(rate=0.3)(dense)



    #out

    head_root = Dense(168, activation = 'softmax',name="roots")(dense)

    head_vowel = Dense(11, activation = 'softmax',name="vowels")(dense)

    head_consonant = Dense(7, activation = 'softmax',name="consonants")(dense)



    model = Model(inputs=inp, outputs=[head_root, head_vowel, head_consonant])

    

    return model







def load_img_data(path,img_size):

    

    df=pd.read_parquet(path,engine="pyarrow")

    df=df.drop(["image_id"],axis=1)

    a=df.values

    a.astype("float16")

    a=np.reshape( a,(a.shape[0],)+(137,236))

    s=a.shape[0]

    

    

    A=[]

    for i in a:

        A.append(list(cv2.resize(i,(img_size,img_size))))

    A=np.array(A)/255

    

    X_train=A.reshape(-1,img_size,img_size,1)

    

    print("IMG Data Created, Shape:", A.shape)

    

    del df

    del a

    gc.collect()

    

    return X_train,s





###----train-----







#variables

img_size=74

BATCH_SIZE=84

lepochs=[10,30,30,80]



#NAME="WTFWTF{}".format(int(time.time()))

#tensorboard= TensorBoard(log_dir='logs/{}'.format(NAME))





inp=Input(shape=(img_size,img_size,1))

model=SimpleNet(inp)





adamm=keras.optimizers.Adam(learning_rate=0.0003, beta_1=0.9, beta_2=0.999, amsgrad=False)



#lrr_root = ReduceLROnPlateau(monitor='roots_accuracy',patience=2,verbose=1,factor=0.5,cooldown=3,min_lr=0.000005)

#callbacks= [lrr_root]



#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.compile(optimizer=adamm, loss='categorical_crossentropy', metrics=['accuracy'])

    



model.summary()







t1=time.time()

sp=0

histories=[]

for i in [0,1,2,3]:

    

    epochs=lepochs[i]

    print("Begin Loading Data Set:",i)

    

    #GET DATA

    path="/kaggle/input/bengaliai-cv19/train_image_data_"+str(i)+".parquet"

    X_train, L = load_img_data(path,img_size=img_size)

    

    train_df=pd.read_csv("/kaggle/input/bengaliai-cv19/train.csv")

    root = pd.get_dummies(train_df['grapheme_root']).values[sp:sp+L]

    vowel = pd.get_dummies(train_df['vowel_diacritic']).values[sp:sp+L]

    consonant = pd.get_dummies(train_df['consonant_diacritic']).values[sp:sp+L]

    

    print("Label Data Created, Shape:", root.shape, consonant.shape, vowel.shape)

    

    

    x_train, x_test, y_train_root, y_test_root, y_train_vowel, y_test_vowel, y_train_consonant, y_test_consonant = train_test_split(X_train, root, vowel, consonant, test_size=0.08, random_state=666)

    sp=sp+L

    

    

    if i==3:

        model.optimizer.optimizer.lr = 0.0002

        

#     if i==2:

#         model.optimizer.optimizer.lr = 0.0000125

    

#     if i==3:

#         model.optimizer.optimizer.lr = 0.0000075



#     print(model.optimizer.optimizer.lr)

    

    #SETUP

    datagen = MultiOutputDataGenerator(featurewise_center=False,samplewise_center=False,featurewise_std_normalization=False,samplewise_std_normalization=False,zca_whitening=False,rotation_range=8,zoom_range = 0.15,width_shift_range=0.15,height_shift_range=0.15,horizontal_flip=False,vertical_flip=False)

    datagen.fit(x_train)



    #RUN

    

    print("Start Training Iter:",i)

    history = model.fit_generator(datagen.flow(x_train, {'roots': y_train_root, 'vowels': y_train_vowel, 'consonants': y_train_consonant}, batch_size=BATCH_SIZE),epochs = epochs, validation_data = (x_test, [y_test_root, y_test_vowel, y_test_consonant]),steps_per_epoch=x_train.shape[0] // BATCH_SIZE)

    

    print(history)

    

    del train_df

    del root

    del vowel

    del consonant

    del x_train

    del x_test

    del y_test_root

    del y_train_vowel

    del y_test_vowel

    del y_train_consonant

    del y_test_consonant

    gc.collect()

    

    h1=history.history["roots_accuracy"]

    v1=history.history["roots_loss"]

    histories.append(h1)

    plt.plot(h1)

    plt.show()

    

    plt.plot(v1)

    plt.show()

    

#     plt.plot(list(range(len(histories))),histories)

#     plt.show()

    

print("Done!")

print(time.time() - t1)







# histories=history.history["roots_accuracy"]

# h0,h1,h2,h3 = histories[0:epochs], histories[epochs:epochs*2], histories[epochs*2:epochs*3], histories[epochs*3:epochs*4]



# plt.plot(list(range(0,epochs)), h0, color="b") 

# plt.plot(list(range(epochs,epochs*2)), h1, color="r")

# plt.plot(list(range(epochs*2,epochs*3)), h2, color="b")

# plt.plot(list(range(epochs*3,epochs*4)), h3, color="g")



# plt.show()
h1=history.history["roots_accuracy"]

v1=history.history["roots_loss"]



print("Max", h1[-1])

print(" ")

plt.plot(h1)

plt.show()



plt.plot(v1)

plt.show()

    



preds_dict = {'grapheme_root': [], 'vowel_diacritic': [], 'consonant_diacritic': []}

    

components = ['consonant_diacritic', 'grapheme_root', 'vowel_diacritic']

target,row_id=[],[]



img_size=74

for i in range(4):

    

    path='/kaggle/input/bengaliai-cv19/test_image_data_{}.parquet'.format(i)

    X_test,L = load_img_data(path,img_size)

    

    preds = model.predict(X_test)

    #print(preds)

    print("Getting df again...")

    df_test_img = pd.read_parquet('/kaggle/input/bengaliai-cv19/test_image_data_{}.parquet'.format(i))

    df_test_img.set_index('image_id', inplace=True)

    

    print(df_test_img.head())

    

    for i, p in enumerate(preds_dict):

        preds_dict[p] = np.argmax(preds[i], axis=1)



    for k,id in enumerate(df_test_img.index.values):  

        for i,comp in enumerate(components):

            id_sample=id+'_'+comp

            row_id.append(id_sample)

            target.append(preds_dict[comp][k])



    del df_test_img

    del X_test

    gc.collect()



df_sample = pd.DataFrame(

    {

        'row_id': row_id,

        'target':target

    },

    columns = ['row_id','target'] 

)

df_sample.to_csv('submission.csv',index=False)

print(df_sample.head())



print("Time", time.time() - t1)