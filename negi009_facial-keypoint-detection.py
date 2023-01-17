# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
print(os.listdir("../input/training"))

print(os.listdir("../input/test"))
train = pd.read_csv('../input/training/training.csv')
train.head().T
#train['Image'][0]

print('size of traning data {}'.format(len(train)))

print('Missing vlaue col ')

print(train.isnull().any().value_counts())

train.isnull().sum().sort_values(ascending=False)
#ffill which fills the place with value in the Forward index or Previous/Back respectively.



train.fillna(method='ffill',inplace=True)

#check missig col

train.isnull().any().value_counts()
len(train)
# convert image col to int  also check NaN

image_list=[]



for i in train['Image']:

    i=i.split(' ')

    image_list.append(i)

len(image_list)



len(image_list[0])

#covert to arry

image_list = np.array(object=image_list,dtype=float)
images=image_list.reshape(-1,96,96,1)
plt.imshow(image_list.reshape(-1,96,96)[3010],cmap='gray')



IMAGE_HEIGHT=96

IMAGE_WIDTH=96

#Ytrain.iloc[3010][Ytrain.iloc[3010].index[1]]

#np.dtype(Ytrain.iloc[3010]['left_eye_center_y'])

def img_show(image_list,train):

    fig,axes = plt.subplots(nrows=5,ncols=2,dpi=300,figsize=(12,12))



    for row in range(5):

        for col in range(1):

            #random number  generator for diff image

            j  =np.random.randint(0,len(train))

            X = image_list.reshape(-1,96,96)[j]

            Y = train



            Y=Y.iloc[j]# location of Y

            img = np.copy(X) #copy image

            for i in range(0,30,2):

            #print(Y[Y.index[i+1]])

                   if 0 < Y[Y.index[i]] < IMAGE_WIDTH and  0 < Y[Y.index[i+1]] < IMAGE_HEIGHT:

                    img[int(Y[Y.index[i+1]]),int(Y[Y.index[i]])] = 255

            axes[row,col].imshow(img,cmap='gray')

            axes[row,col+1].imshow(X,cmap='gray')

            #remove axies

            axes[row,col].axis('off')

            axes[row,col+1].axis('off')



    plt.tight_layout()

img_show(image_list,train.drop(labels='Image',axis=1))
y_train=train.drop(labels='Image',axis=1)

y_train.shape
X_train=images

X_train.shape

#lenght of tensor has 4 index
X_train=X_train/255

X_train[1]


import tensorflow as tf

model= tf.keras.models.Sequential(

    

    layers=[

        

         #convolution 1st time

        tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),activation=tf.nn.relu,input_shape=(96,96,1)),

        tf.keras.layers.MaxPool2D(2,2),

     

         #convolution 2nd time

        tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),activation=tf.nn.relu,input_shape=(96,96,1)),

        tf.keras.layers.MaxPool2D(2,2),

       

         #convolution 2nd time

        tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),activation=tf.nn.relu,input_shape=(96,96,1)),

        tf.keras.layers.MaxPool2D(2,2),

      

        #input layer

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(units=526,activation='relu'),

        tf.keras.layers.Dense(units=526,activation='relu'),

        tf.keras.layers.Dropout(0.3),

        



        # number of keypoint

        tf.keras.layers.Dense(units=30,activation='relu')

    ]

)
model.summary()
model.compile(optimizer='adam',

             loss='mse',

              metrics=['acc'])
hist=model.fit(x=X_train,y=y_train,batch_size=128,epochs=200,verbose=2,validation_split=0.2)

hist
from sklearn.metrics import r2_score



y_pred =model.predict(X_train)



score = r2_score(y_train,y_pred)

score


plt.figure(figsize=(20,10))

loss=hist.history['loss']

val_loss=hist.history['val_loss']

y=np.arange(1,201)

plt.plot(y,loss,'b',label='train')

plt.plot(y,val_loss,'r',label='val')

plt.xlabel('epoch')

plt.ylabel('loss')



plt.legend()



plt.legend()
test = pd.read_csv('../input/test/test.csv')
y=np.arange(1,501)
# convert image col to int  also check NaN

image_list=[]



for i in test['Image']:

    i=i.split(' ')

    image_list.append(i)

len(image_list)
image_list=np.array(image_list,dtype=float)

images=image_list.reshape(-1,96,96,1)

X_test =images/255.0

predicted_value =model.predict(X_test)
pv =pd.DataFrame(data=predicted_value)

img_show(image_list,pv)
pred = model.predict(X_test)

lookid_data = pd.read_csv('../input/IdLookupTable.csv')

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

submission.to_csv('submision.csv',index = False)
df=pd.read_csv('submision.csv')
df