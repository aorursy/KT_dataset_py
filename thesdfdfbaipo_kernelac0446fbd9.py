import numpy as np
import pandas as pd
import os
import math
import sys
from scipy import misc
import tensorflow as tf
from skimage import img_as_float
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint
from keras.layers import Dense,Dropout,Conv2D,MaxPooling2D,BatchNormalization,Input,UpSampling2D,Concatenate,Activation,Add,Flatten,Concatenate,Lambda,Reshape,AveragePooling2D
from keras.models import Model
from keras.models import load_model
from keras.models import model_from_json
from keras.optimizers import Adam
import keras.metrics
import matplotlib.pyplot as plt
from keras import backend as K
#from keras import Callback
from sklearn.model_selection import train_test_split
config = tf.ConfigProto()
jit_level = tf.OptimizerOptions.ON_1
config.graph_options.optimizer_options.global_jit_level = jit_level
sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)
name_label_dict = {
0:  'Nucleoplasm',
1:  'Nuclear membrane',
2:  'Nucleoli',
3:  'Nucleoli fibrillar center',
4:  'Nuclear speckles',
5:  'Nuclear bodies',
6:  'Endoplasmic reticulum',
7:  'Golgi apparatus',
8:  'Peroxisomes',
9:  'Endosomes',
10:  'Lysosomes',
11:  'Intermediate filaments',
12:  'Actin filaments',
13:  'Focal adhesion sites',   
14:  'Microtubules',
15:  'Microtubule ends',  
16:  'Cytokinetic bridge',   
17:  'Mitotic spindle',
18:  'Microtubule organizing center',  
19:  'Centrosome',
20:  'Lipid droplets',
21:  'Plasma membrane',   
22:  'Cell junctions', 
23:  'Mitochondria',
24:  'Aggresome',
25:  'Cytosol',
26:  'Cytoplasmic bodies',   
27:  'Rods & rings' }
PATH = '../input/'
TRAIN = '../input/dadada/tdd/train/'
TEST = '../input/human-protein-atlas-image-classification/test/'
LABEL = '../input/ce-unet-1/nntrain.csv'
SAMPLE = '../input/human-protein-atlas-image-classification/sample_submission.csv'
label = pd.read_csv(LABEL)
train_names = list(set('_'.join((f.split('_'))[:-1]) for f in os.listdir(TRAIN)))
test_names = list(set('_'.join((f.split('_'))[:-1]) for f in os.listdir(TEST)))
train_data, valid_data = train_test_split(train_names, test_size=0.1, random_state=42)
def open_rgby(path,pid): #a function that reads RGBY image
    colors = ['red','green','blue','yellow']
    tmp = np.zeros(shape=(512,512,4))
    for i,color in enumerate(colors) :
        tmp[:,:,i] = img_as_float(misc.imread(os.path.join(path, str(pid)+'_'+color+'.png')))
    return tmp
def generator(train,label,batch_size) :
    batch_features = np.zeros((batch_size,512,512,4))
    batch_labels = np.zeros((batch_size,28))
    while True :
        for i in range(batch_size) :
            idx= np.random.choice(len(train),1)[0]
            batch_features[i,:,:,:] = open_rgby(TRAIN,train[idx])
            temp = set([int(i) for i in (label.loc[label['Id']==train[idx]]['Target'].values[0]).split()])
            batch_labels[i,:] =  np.array([1 if i in temp else 0 for i in range(28)])
        
        out = {'main':batch_labels,'encoder':batch_features}
        yield batch_features,out
def progressbar(name,i,n):    
    sys.stdout.write('\r')
    sys.stdout.write(name+": [%-20s] %d%% %d/%d" % ('=' * int(20. * (i + 1) / n - 1) + '>',
                                                    int(100. * (i + 1) / n) , i,n))
    sys.stdout.flush()
def padding(tensor,h,w,r):
    return tf.pad(tensor, tf.constant([[0,0],[h,h], [w,w],[r,r]]), "CONSTANT")

model = load_model('../input/ce-unet-2/ce_Unet_save28.h5')
epoch = 4
history = model.fit_generator(generator(train_data,label,4), 
                                  samples_per_epoch=len(train_data)//4, epochs=epoch,
                                  validation_data=generator(valid_data,label,4),
                                  validation_steps=len(valid_data)//4,
                                  callbacks=[ModelCheckpoint('mymodel_valid.h5',monitor='val_loss',save_best_only=True),
                                  ReduceLROnPlateau(monitor='val_loss',patience=3)])
model.save('ce_Unet_save32.h5')
model.save('ce_Unet_save32.h5')
plt.plot(history.history['main_loss'])
plt.plot(history.history['val_main_loss'])
plt.savefig("ce_Unet_loss_history.png")