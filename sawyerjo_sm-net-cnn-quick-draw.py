import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import ast

import os

from glob import glob

from tqdm import tqdm

from dask import bag

import cv2

import tensorflow as tf

from tensorflow import keras

from keras.models import Sequential

from keras.layers import Dense, Activation, DepthwiseConv2D, BatchNormalization, ZeroPadding2D

from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, AveragePooling2D 

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from keras.metrics import top_k_categorical_accuracy

import os

from tqdm import tqdm #for문 처리과정 확인 함수 

path = os.listdir('/kaggle/input/quickdraw-doodle-recognition/train_simplified')

path

df = pd.read_csv('/kaggle/input/quickdraw-doodle-recognition/train_simplified/'+path[0])

df
def stroke_to_img(strokes): 

    img=np.zeros((256,256))

    for each in ast.literal_eval(strokes):

        for i in range(len(each[0])-1):

            cv2.line(img,(each[0][i],each[1][i]),(each[0][i+1],each[1][i+1]),255,5)

    img=cv2.resize(img,(32,32))

    img=img/255

    return img



tmp = df['drawing'][8]

df['word']= df['word'].replace(' ','_',regex = True)

img = np.array(stroke_to_img(tmp))

plt.imshow(img)

print(img)
rd=np.random.randint(340)#340까지의 수 중에 난수생성

nums2names={i : v[:-4].replace(' ','_') for i , v in enumerate(path)}#불러온 데이터에 index 번호붙이기

ranclass=nums2names[rd]# 340까지의 수중에 랜덤으로 불러오기  

ranclass=ranclass.replace('_',' ')# 분류 제목을 _을 ' '로 바꿔주기

rdpath='/kaggle/input/quickdraw-doodle-recognition/train_simplified/'+ranclass+'.csv' #랜덤으로 하나의 클래스 경로설정

one=pd.read_csv(rdpath,usecols=['drawing','recognized','word'],nrows=10) #10개 행의 drawing recognized word 불러오기

one=one[one.recognized==True].head(2)#그림 중 true인것 2개만불러오기

name=one['word'].head(1)#one의 첫번째 word 

strk=one['drawing']# one 의 drawing #2개

pic=[]

for s in strk:

    pic.append(stroke_to_img(s))

    #pic에 그려준것 추가하기

name=name.values

train_grand=[]

num_class = 340

per_class=2000
class_paths = glob('/kaggle/input/quickdraw-doodle-recognition/train_simplified/*.csv')

for i , c in enumerate(tqdm(class_paths[0:num_class])): 

    train=pd.read_csv(c,usecols=['drawing','recognized'],nrows=per_class*2)

    train=train[train.recognized==True].head(per_class)

    imagebag=bag.from_sequence(train.drawing.values).map(stroke_to_img)

    train_array=np.array(imagebag.compute())#unmpy 형식 

    train_array=np.reshape(train_array,(per_class,-1))  #2000  -1로 reshpae로 행렬 형식 변경 그림형식    

    label_array=np.full((train.shape[0],1),i)# label 붙여주기 , train.shape[0] = 2000

    train_array=np.concatenate((label_array,train_array),axis=1)

    train_grand.append(train_array)

del train_array

del label_array

train_grand=np.array([train_grand.pop() for i in np.arange(num_class)]) #데이터 세팅
height = 32

width = 32
train_grand=train_grand.reshape((-1,(height*width+1))) #32*32로 배열 변경

print(train_grand)
specific = 0.1 

sequence_length = 50

cut = int(specific * train_grand.shape[0])

print(cut)



np.random.shuffle(train_grand)

y_train, X_train = train_grand[cut: , 0], train_grand[cut: , 1:]

y_val, X_val = train_grand[0:cut, 0], train_grand[0:cut, 1:]



# del train_grand



x_train=X_train.reshape(X_train.shape[0],height,width,1)

x_val=X_val.reshape(X_val.shape[0],height,width,1)



print(y_train.shape, "\n",

      x_train.shape, "\n",

      y_val.shape, "\n",

      x_val.shape)



model = Sequential()

model.add(Conv2D(32,kernel_size=(3,3),padding='same',activation='relu',input_shape=(height,width,1)))

# 32*32*32

model.add(MaxPooling2D(pool_size=(2,2))) 

# 16*16*32

model.add(Conv2D(64,kernel_size=(3,3),padding='valid',activation='relu'))

# 12*12*64

model.add(MaxPooling2D(pool_size=(2,2))) 

# 6*6*64

model.add(DepthwiseConv2D(64, padding='same',activation='relu' ))

# 6*6*64

model.add(MaxPooling2D(pool_size=(2,2))) 



model.add(BatchNormalization())

# 6*6*64

model.add(Activation('relu'))



model.add(ZeroPadding2D(padding=(1, 1)))



model.add(Conv2D(128,kernel_size=(3,3),padding='same',activation='relu'))

# 6*6*128

model.add(AveragePooling2D(pool_size=(2,2)))

# 3*3*128

# 3*3*128



# 5*5*128

model.add(Conv2D(256,kernel_size=(3,3),padding='same',activation='relu'))

# 5*5*256

model.add(Conv2D(256,kernel_size=(3,3),padding='same',activation='relu'))

# 5*5*256

model.add(Flatten())

# 2560

model.add(Dense(num_class*5, activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(num_class,activation='softmax'))

model.summary()



def top_3_accuracy(y_true, y_pred):

    return top_k_categorical_accuracy(y_true, y_pred, k=3)
reduceLROnPlat=ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=3,

                                 verbose=1,mode='auto',min_delta=0.005,

                                 cooldown=5,min_lr=0.0001)



callbacks=[reduceLROnPlat]



model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',

              metrics=['accuracy',top_3_accuracy])



history=model.fit(x=x_train,y=y_train,batch_size=32,epochs=20,

                  validation_data=(x_val,y_val),callbacks=callbacks,verbose=1)
acc=history.history['accuracy']

val_acc=history.history['val_accuracy']

loss= history.history['loss']

val_loss=history.history['val_loss']



epochs=range(1,len(acc)+1)



plt.plot(epochs,acc,label='Training acc')

plt.plot(epochs,val_acc,label='Validation acc')

plt.title('Training and validation accuracy')

plt.legend()



plt.figure()



plt.plot(epochs,loss,label='Training loss')

plt.plot(epochs,val_loss,label='Validation loss')

plt.title('Training and validation loss')

plt.legend()



plt.show()
list=[]

reader=pd.read_csv('/kaggle/input/quickdraw-doodle-recognition/test_simplified.csv',index_col=['key_id'],chunksize=2048)

for chunk in tqdm(reader,total=55):

    imagebag=bag.from_sequence(chunk.drawing.values).map(stroke_to_img)

    testarray=np.array(imagebag.compute())

    testarray=np.reshape(testarray,(testarray.shape[0],height,width,1))

    testpreds=model.predict(testarray,verbose=0)

    s=np.argsort(-testpreds)[:,0:3]

    list.append(s)

array=np.concatenate(list)

pred_df=pd.DataFrame({'first': array[:,0],'second':array[:,1],'third':array[:,2]})

pred_df=pred_df.replace(nums2names)

pred_df['words']=pred_df['first']+' '+pred_df['second']+' '+pred_df['third']



sub=pd.read_csv('/kaggle/input/quickdraw-doodle-recognition/sample_submission.csv',index_col=['key_id'])

sub['word']=pred_df.words.values

sub.to_csv('result_of_mission.csv')
sub.head()