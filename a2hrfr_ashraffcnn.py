# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
DATA_PATH = '../input/bengaliai-cv19/'

train_labels = pd.read_csv(DATA_PATH + 'train.csv')

test_labels = pd.read_csv(DATA_PATH + 'test.csv')

class_map = pd.read_csv(DATA_PATH + 'class_map.csv')

sample_submission = pd.read_csv(DATA_PATH + 'sample_submission.csv')
train_labels.head()
for col in train_labels:

    print(col , '----> unique values = ' , train_labels[col].unique().shape[0])
for col in class_map:

    print(col , '----> unique values = ' , class_map[col].unique().shape[0])
import matplotlib.pyplot as plt

import numpy as np
def by_cat(CC,GG):

    con=train_labels[CC].unique()

    all_cols=train_labels.drop(['image_id',CC,GG],axis=1).columns

    fig,ax=plt.subplots(len(con),len(all_cols),figsize=(20,15))

    for i in range(len(con)):

        temp=train_labels[train_labels[CC]==con[i]].groupby(GG).count()

        for c in range(len(all_cols)):

            if all_cols[c] in temp:

                ax[i,c].bar(temp.index,temp[all_cols[c]])

                ax[i,c].set_title(str(all_cols[c]) + ' ' + str(con[i]))

    plt.tight_layout()

    plt.show()

    



   



    

    
by_cat('consonant_diacritic','vowel_diacritic')

by_cat('consonant_diacritic','grapheme_root')

len(train_labels)
train_df_0 = pd.read_parquet(DATA_PATH + 'train_image_data_0.parquet')

train_df_0.head()
import matplotlib.pyplot as plt

import numpy as np



H,W = (137,236)

 

def plot_img(val):

    img = val.reshape(H, W)

    plt.imshow(img.astype(float), cmap='gray')

    plt.show()

    return img.astype(float)



def crop_img(img_,th):

    arr=np.argwhere(img>th)

    x=arr[:,0]

    y=arr[:,1]  

    fRow = min(x)

    lRow = max(x)

    fCol = min(y)

    lCol = max(y)

    return img[fRow:lRow,fCol:lCol]





def setup_img(val,th):

    val=val/val.max()

    img = val.reshape(H, W)

    img=1-img

    img[img<th]=0

    return img.astype(float)



def apply_th(img,th):

    img[img<th]=0

    img[img>=th]=1

    return img

    
fig,ax=plt.subplots(1,3,figsize=(10,10))





img=setup_img(train_df_0.iloc[2].values[1:],0.1)

ax[0].imshow(img,cmap='gray')

ax[0].set_title('orginal invert')



corped_img=crop_img(img,0.5)

ax[1].imshow(corped_img,cmap='gray')

ax[1].set_title('croped img')





th_img=apply_th(corped_img,0.3)

ax[2].imshow(th_img,cmap='gray')

ax[2].set_title('threshold')



plt.show()
#let see the avg img size 
HH=[]

WW=[]

for i in range (0,100):

    img=setup_img(train_df_0.iloc[i].values[1:],0.1)

    corped_img=crop_img(img,0.5)

    HH.append(corped_img.shape[0])

    WW.append(corped_img.shape[1])
from scipy import stats
np.mean(HH),np.mean(WW)
stats.mode(HH),stats.mode(WW)
# LETS TRY 80 X 110
import cv2
#corped_img

img_resized= cv2.resize(corped_img, dsize=(80, 110), interpolation=cv2.INTER_CUBIC      )

plt.imshow(img_resized,cmap='gray')
#th_img

img_resized= cv2.resize(th_img, dsize=(80, 110), interpolation=cv2.INTER_CUBIC)

plt.imshow(img_resized,cmap='gray')
for i in range(10):

    fig,ax=plt.subplots(1,2,figsize=(10,10))

    img1=PreprocessImg.get_img(train_df_0.iloc[i].values[1:],True,True)

    img2=PreprocessImg.get_img(train_df_0.iloc[i].values[1:],True,False)



    ax[0].imshow(img1,cmap='gray')

    ax[1].imshow(img2,cmap='gray')

    plt.show()
import pandas  as pd

import cv2

from  tqdm import tqdm



def crop_and_resize_images(df, resized_df, resize_size = 80):

    cropped_imgs = {}

    for img_id in tqdm(range(df.shape[0])):

        img = resized_df[img_id]

        _, thresh = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]

        

        idx = 0 

        ls_xmin = []

        ls_ymin = []

        ls_xmax = []

        ls_ymax = []

        for cnt in contours:

            idx += 1

            x,y,w,h = cv2.boundingRect(cnt)

            ls_xmin.append(x)

            ls_ymin.append(y)

            ls_xmax.append(x + w)

            ls_ymax.append(y + h)

        xmin = min(ls_xmin)

        ymin = min(ls_ymin)

        xmax = max(ls_xmax)

        ymax = max(ls_ymax)



        roi = img[ymin:ymax,xmin:xmax]

        resized_roi = cv2.resize(roi, (resize_size, resize_size))

        resized_roi=resized_roi/255

        cropped_imgs[df.image_id[img_id]] = resized_roi.reshape(-1)

        

    resized = pd.DataFrame(cropped_imgs).T.reset_index()

    resized.columns = resized.columns.astype(str)

    resized.rename(columns={'index':'image_id'},inplace=True)

    return resized #out_df
TRAIN=['train_image_data_0.parquet','train_image_data_1.parquet','train_image_data_2.parquet','train_image_data_3.parquet']

DATA_PATH = '../input/bengaliai-cv19/'

HEIGHT = 137

WIDTH = 236



for i,dName in enumerate(TRAIN):

    df = pd.read_parquet(DATA_PATH + dName )

    resized = df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH)

    name = '80x80_train_{}.feather'.format(i)

    cropped_df = crop_and_resize_images(df,resized)

    cropped_df.to_feather(name)

    del df,resized,cropped_df



    

    
'''

grapheme_root ----> unique values =  168

vowel_diacritic ----> unique values =  11

consonant_diacritic ----> unique values =  7

'''
import cv2

#shape=(80,80)

H,W = (137,236)

th=0.3

class PreprocessImg():

    def get_img(val,shape,apply_th=False,resize=True):

        img_=PreprocessImg.setup_img(val)

        img_=PreprocessImg.crop_img(img_,th)

        if apply_th:

            img_=PreprocessImg.apply_th(img_,th)

        if resize:

            img_= cv2.resize(img_, dsize=shape, interpolation=cv2.INTER_CUBIC)

        return img_

        

    def crop_img(img,th):

        arr=np.argwhere(img>=th)

        x=arr[:,0]

        y=arr[:,1]  

        fRow = min(x)

        lRow = max(x)

        fCol = min(y)

        lCol = max(y)

        return img[fRow:lRow,fCol:lCol]



    def setup_img(val):

        val=val/val.max()

        img = val.reshape(H, W)

        img=1-img

        return img.astype(float)

    

    def apply_th(img,th):

        img[img<th]=0

        img[img>=th]=1

        return img
def one_hot_encoding(c):

    arr=np.zeros((168))

    arr[c]=1

    return arr



def x_y(x,y):

    y_arr=np.zeros((25,168),dtype=np.float32)

    for c in range(25):

        y_arr[c] = one_hot_encoding(y[c])

        

    x_arr=np.zeros((x.shape[0],80,80,1),dtype=np.float32)

    for c in range((x.shape[0])):

        im=PreprocessImg.get_img(x[c],True,True)

        x_arr[c] = np.reshape(im,(80,80,1))

    return x_arr,y_arr
import pandas as pd

import numpy as np

import time    

import cv2

from tensorflow.keras.utils import Sequence



TRAIN_RANGE=200840

TEST_RANGE=12

BATCH_SIZE=25

CLASS_=168

IMAGE_SIZE=(80,80)

DATA_PATH = '../input/bengaliai-cv19/'

TRAIN=['train_image_data_0.parquet','train_image_data_1.parquet','train_image_data_2.parquet','train_image_data_3.parquet']

TEST=['test_image_data_0.parquet','test_image_data_1.parquet','test_image_data_2.parquet','test_image_data_3.parquet',]





class batch():



    def __init__(self,batch_size,to_fit=True):

        self.batch_size=batch_size

        self.pre=0

        self.to_fit=to_fit

        if to_fit:

            self.data=pd.read_parquet(DATA_PATH + TRAIN[0])

            self.lis_of_file=TRAIN

            self.LABLES=pd.read_csv(DATA_PATH +  'train.csv')

        else:

            self.data=pd.read_parquet(DATA_PATH + TEST[0])

            self.lis_of_file=TEST

            

        self.len=len(self.data)

        

        self.index=0



    def set_next_df(self):

        if self.index + 1 < len(self.lis_of_file):

            self.data=pd.read_parquet(DATA_PATH + self.lis_of_file[self.index + 1])

            self.index += 1

        else:

            self.data=pd.read_parquet(DATA_PATH + self.lis_of_file[0])

            self.index = 0

        self.pre=0

        self.len=len(self.data)





    def next_batch(self,get_as_array=True):

        if self.pre +  self.batch_size > self.len:

            if self.pre < self.len:

                x=self.data.iloc[self.pre  : ]

                len_of_x= len(x)

                self.set_next_df()

                x=x.append(self.data.iloc[self.pre : self.batch_size - len_of_x ] , ignore_index=True)

                self.pre = self.batch_size - len_of_x

                y=self.get_y(x)

                if get_as_array:

                    return self.get_asArray(x,y)

                return x,y

            else:

                    self.set_next_df

        x=self.data.iloc[self.pre  :  self.pre +  self.batch_size]

        y=self.get_y(x)

        self.pre+=self.batch_size

        if get_as_array:

            return self.get_asArray(x,y)

        return x,y



    def get_y(self,data):

        if  self.to_fit == False:

            return []

        l=len(data)

        d=data['image_id'].iloc[0]

        index=self.LABLES[self.LABLES['image_id']==d].index[0]

        if index + l > TRAIN_RANGE:

            y=self.LABLES.iloc[index:]

            len_of_y= len(y)

            y=y.append(self.LABLES[0: l - len_of_y ])

            return y

        y=self.LABLES.iloc[index:index+l]

        return y

    

    def get_asArray(self,x,y):

        y_val=[]

        if self.to_fit == True :

            y_val=y.drop(['image_id','grapheme','vowel_diacritic','consonant_diacritic'],axis=1).values

        x_val=x.drop('image_id',axis=1).values

        #y_val=y.drop(['image_id','grapheme'],axis=1).values

        return x_val,y_val





class DataGenerator(Sequence):

    """Generates data for Keras

    Sequence based data generator. Suitable for building data generator for training and prediction.

    """

    def __init__(self,to_fit=True, batch_size=25, dim=(80, 80), n_classes=168,final_=True):

     

        self.to_fit = to_fit

        self.batch_size = batch_size

        self.dim = dim

        self.n_classes = n_classes

        self.b=batch(batch_size = batch_size , to_fit = to_fit)

        self.final_=final_

        #self.on_epoch_end()



    def __len__(self):

        

        return int(np.floor(TRAIN_RANGE / self.batch_size))



    def __getitem__(self, index):

        

       

        X,y=self.b.next_batch()

        if  self.final_: 

            X,y=self.FinalXY(X,y)



        if self.to_fit:

            return X, y

        else:

            return X

        

    def one_hot_encoding(self,c):

        arr=np.zeros((CLASS_))

        arr[c]=1

        return arr



    def FinalXY(self,x,y):

        y_arr=[]

        if self.to_fit:

            y_arr=np.zeros((self.batch_size,CLASS_),dtype=np.float32)

            for c in range(self.batch_size):

                y_arr[c] = self.one_hot_encoding(y[c])

            

        x_arr=np.zeros((x.shape[0],IMAGE_SIZE[0],IMAGE_SIZE[1],1),dtype=np.float32)

        for c in range((BATCH_SIZE)):

            im=PreprocessImg.get_img(x[c],IMAGE_SIZE,True,True)

            x_arr[c] = np.reshape(im,(IMAGE_SIZE[0],IMAGE_SIZE[1],1))

            

        return x_arr,y_arr




from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense , Activation , Dropout , Flatten ,Conv2D , MaxPool2D ,BatchNormalization 

from tensorflow.keras import optimizers

def setup_model(input_shape):

    model = Sequential()

    model.add(Conv2D(64,(3,3),input_shape=input_shape,activation='relu',padding='same'))

    model.add(Conv2D(64,(3,3),activation='relu',padding='same'))

    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Dropout(0.2))



    model.add(Conv2D(128,(3,3),activation='relu',padding='same'))

    model.add(Conv2D(64,(3,3),activation='relu',padding='same'))

    model.add(Conv2D(32,(3,3),activation='relu',padding='same'))

    model.add(Dropout(0.2))



    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Dropout(0.2))



    model.add(Conv2D(128,(3,3),activation='relu',padding='same'))

    model.add(Conv2D(64,(3,3),activation='relu',padding='same'))

    model.add(Conv2D(32,(3,3),activation='relu',padding='same'))

    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Dropout(0.2))



    model.add(Conv2D(128,(3,3),activation='relu',padding='same'))

    model.add(Conv2D(64,(3,3),activation='relu',padding='same'))

    model.add(Conv2D(32,(3,3),activation='relu',padding='same'))

    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Dropout(0.2))



    model.add(Flatten())

    model.add(Dense(256))

    model.add(Dense(128))

    model.add(Dense(64))



    #model.add(Activation('relu'))

    model.add(Dropout(0.2))

    model.add(Dense(168,activation='softmax'))



    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],optimizer='adam')

    print(model.summary())

    return model
model=setup_model((80,80,1))
mname='Mbangi'
import keras

class CustomSaver(keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):

        if epoch % 10 == 0 : 

            self.model.save("{}_{}.hd5".format(mname,epoch))
Gen = DataGenerator(to_fit=True, batch_size=BATCH_SIZE, dim=(80, 80), n_classes=168,final_=True)
saver = CustomSaver()
hist=model.fit_generator(generator=Gen,epochs=20,verbose=1,shuffle=True,callbacks=[saver])
BATCH_SIZE