!pip install ../input/kerasapplications/keras-team-keras-applications-3b180cb -f ./ --no-index

!pip install ../input/efficientnet/efficientnet-1.1.0/ -f ./ --no-index
import efficientnet.tfkeras as efn
list(range(5))

import os

import pydicom

import re

import cv2

import math

import time

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm_notebook as tqdm

from sklearn.model_selection import train_test_split

from kaggle_datasets import KaggleDatasets



import tensorflow as tf

import tensorflow.keras.backend as K

from tensorflow.keras.layers import Dense

from tensorflow.keras.models import Model,Sequential

from tensorflow.keras import optimizers

import efficientnet.tfkeras as efn

from tensorflow.keras.utils import Sequence

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import KFold,GroupKFold
import random

def seed_everything(seed=2020):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    tf.random.set_seed(seed)

    

seed_everything(42)
ROOT = "../input/osic-pulmonary-fibrosis-progression"

tr = pd.read_csv(f"{ROOT}/train.csv")

tr.drop_duplicates(keep=False, inplace=True, subset=['Patient','Weeks'])
data=tr

data['min_week'] = data['Weeks']

data['min_week'] = data.groupby('Patient')['min_week'].transform('min')

base = data.loc[data.Weeks == data.min_week].copy()

base['normal_FVC']=base.FVC/base.Percent*100



base = base[['Patient','FVC','normal_FVC','Percent']].copy()

base.columns = ['Patient','min_FVC','normal_FVC','base_percent']

base['nb'] = 1

base['nb'] = base.groupby('Patient')['nb'].transform('cumsum')

base = base[base.nb==1]

base.drop('nb', axis=1, inplace=True)

data = data.merge(base, on='Patient', how='left')

data['base_week'] = data['Weeks'] - data['min_week']

del base

COLS = ['Sex','SmokingStatus'] #,'Age'

FE = ["Patient"]

for col in COLS:

    for mod in data[col].unique():

        FE.append(mod)

        data[mod] = (data[col] == mod).astype(int)

#data
from sklearn.preprocessing import StandardScaler

num_enc = StandardScaler()

num_cols_to_scale = ['base_week','base_percent','min_FVC','Age',"min_week"]

num_enc.fit(data[num_cols_to_scale])

data[num_cols_to_scale]= num_enc.transform(data[num_cols_to_scale])

train_data=data[["Patient","Male","Female","Ex-smoker","Never smoked","Currently smokes",'base_week','base_percent','min_FVC','Age',"min_week","FVC"]]
BAD_ID = ['ID00011637202177653955184', 'ID00052637202186188008618']

a=train_data.loc[train_data["Patient"]=='ID00011637202177653955184' ]

b=train_data.loc[train_data["Patient"]=='ID00052637202186188008618' ]

#train_data.drop(train_data["Patient"]='ID00011637202177653955184' )

train_data=train_data[(train_data["Patient"]!= 'ID00011637202177653955184') & (train_data["Patient"]!='ID00052637202186188008618' )]

train_data.reset_index(drop=True,inplace=True)

#train_data.shape
def get_img(path):

    d = pydicom.dcmread(path)

    return cv2.resize(d.pixel_array  / 2**11, (384, 384))





# y=train_data.loc[:,"FVC"].values #



# tab=train_data.iloc[:,1:-1].values #



# pred = np.zeros((tab.shape[0], 1)) #
#train_data
sub = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv') 

test = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv') 

#sub
test.rename(columns={"Weeks": "min_week", "FVC": "min_FVC","Percent":"base_percent"},inplace=True)

#test
sub['Patient'] = sub['Patient_Week'].apply(lambda x:x.split('_')[0])

sub['Weeks'] = sub['Patient_Week'].apply(lambda x: int(x.split('_')[-1]))

sub =  sub[['Patient','Weeks','Confidence','Patient_Week']]

sub = sub.merge(test, on="Patient")
sub['base_week'] = sub['Weeks'] - sub['min_week']

#sub
COLS = ['Sex','SmokingStatus'] 



for col in COLS:

    

    for mod in tr[col].unique():

        

        sub[mod] = (sub[col] == mod).astype(int)

#sub
num_cols_to_scale = ['base_week','base_percent','min_FVC','Age',"min_week"]



sub[num_cols_to_scale]= num_enc.transform(sub[num_cols_to_scale])

test_data=sub[["Patient","Male","Female","Ex-smoker","Never smoked","Currently smokes",'base_week','base_percent','min_FVC','Age',"min_week"]]

#test_data
#test_data.iloc[:,1:]
from tensorflow.keras import layers as L

from tensorflow.keras.layers import (

    Dense, Dropout, Activation, Flatten, Input, BatchNormalization, GlobalAveragePooling2D, Add, Conv2D, AveragePooling2D, 

    LeakyReLU, Concatenate 

)
EFN = efn.EfficientNetB4

def build_model(shape=(384, 384, 1)):

    # input layers 

    inp = tf.keras.layers.Input(shape=shape)

    



    base = EFN (input_shape=shape,weights=None,include_top=False)

    x = base(inp)

    x = GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dense(1,activation='relu')(x)

    #x = tf.keras.layers.Dense(1,activation='relu')(x)

    inp2 = tf.keras.layers.Input(shape=(10,))

    x = Concatenate()([x, inp2]) 

    

    x = tf.keras.layers.Dense(64,activation='relu')(x)

    

    #x = tf.keras.layers.Dense(64,activation='relu')(inp2)

    x = tf.keras.layers.Dense(64,activation='relu')(x)

    x = tf.keras.layers.Dense(1,activation='relu')(x)

    model = tf.keras.Model(inputs=[inp,inp2] ,outputs=x)

    opt = tf.keras.optimizers.Adam(learning_rate=0.001)

    loss = tf.keras.losses.MeanAbsoluteError()

    #loss = tf.keras.losses.MeanSquaredError()

    model.compile(optimizer=opt,loss=loss)

    return model
model=build_model()

model.summary()
weight_path="../input/image-1-model-b4"
# fold=0

# print(weight_path + "/fold-{}.h5".format(fold))
#test_data
#BATCH_SIZE=1

BATCH_SIZE=int(len(sub)/5)

BATCH_SIZE
class IGenerator(Sequence):

    def __init__(self, keys, test_data, batch_size=BATCH_SIZE):

        self.keys = [k for k in keys ]

        self.train_data = test_data

        

        self.tab=self.train_data.iloc[:,1:].values

        self.batch_size = batch_size

        



    

    def __len__(self):

        return math.ceil(len(self.keys)/self.batch_size)

    

    def __getitem__(self, index):

        x = []

        a, tab = [], [] 

        key0 = keys[index*self.batch_size:(index+1)*self.batch_size]

        #print(index)

        for k in key0:

            try:  

                

                patient=self.train_data.loc[k,"Patient"]

                

                

                imgs= os.listdir(f'../input/osic-pulmonary-fibrosis-progression/test/{patient}/')

                

                tmp0=math.ceil(len(imgs)/2)-1

                i0=np.random.choice([tmp0-1,tmp0,tmp0+1])

                tmp=[]

                for d in imgs:

                    tmp.append(int(d[:-4]))

                tmp.sort()

                i1=tmp[i0]

                filename =  str(i1) + ".dcm"

                img = get_img(f'../input/osic-pulmonary-fibrosis-progression/test/{patient}/{filename}') 

                x.append(img)

                

                tab.append(self.tab[k])

                



            except:

                print("wrong",k)

       

        x,a,tab = np.array(x), np.array(a), np.array(tab)

        x = np.expand_dims(x, axis=-1)

        return [x, tab]

    

res=[]

for fold in range(5):

    model.load_weights(weight_path + "/fold-{}.h5".format(fold))

    print(fold)

    #for i in range(len(sub)):

#         print(i)

#         patient=test_data.loc[i,"Patient"]

#         imgs= os.listdir(f'../input/osic-pulmonary-fibrosis-progression/test/{patient}/')

#         tmp0=math.ceil(len(imgs)/2)-1

#         i0=np.random.choice([tmp0-1,tmp0,tmp0+1])

#         tmp=[]

#         for d in imgs:

#             tmp.append(int(d[:-4]))

#         tmp.sort()

#         i1=tmp[i0]

#         filename =  str(i1) + ".dcm"

#         img = get_img(f'../input/osic-pulmonary-fibrosis-progression/test/{patient}/{filename}') 

#         tab=test_data.iloc[i,1:].values

#         tab = tab.astype('float64')

#         tab=np.array(tab)

#         img=np.array(img)

#         img = np.expand_dims(img, axis=-1)

        #keys=[i]

    keys=list(range(len(sub)))

    tmp=model.predict(IGenerator(keys,test_data))

    res.append(np.squeeze(tmp))

    

    
#res
#np.array(res).shape
#plt.plot(res)
res1=np.array(res).reshape(5,-1)

#res1.shape
res2=np.mean(res1,0)

#res2.shape
#plt.plot(res2)
sub["FVC"]=res2
sub["Confidence"]=300
#sub
sub1=sub[["Patient_Week","FVC","Confidence"]]

sub1.to_csv("submission.csv", index=False)
#sub1.head(50)