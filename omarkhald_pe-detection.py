# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
    #for filename in filenames:
        #print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install pydicom
from glob import glob
import pydicom
from matplotlib import pyplot as plt
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from tqdm import tqdm_notebook
import tensorflow as tf
train = pd.read_csv("../input/rsna-str-pulmonary-embolism-detection/train.csv")
test = pd.read_csv("../input/rsna-str-pulmonary-embolism-detection/test.csv")
train.head()
train.isna().sum()
train.info()
train.shape
train['filename'] = train[['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID']].apply(
    lambda x: '/'.join(x.astype(str)),
    axis=1
)
train['filename'].head()
from keras.utils import Sequence
from skimage.transform import resize
import math
class generator(Sequence):
    
    def __init__(self,df,images_path,batch_size=32, image_size=256, shuffle=True):
        self.df=df
        self.images_path = images_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.nb_iteration = math.ceil((self.df.shape[0])/self.batch_size)
        self.on_epoch_end()
        
    def load_img(self, filename):
        # load dicom file as numpy array
        img = pydicom.dcmread(filename).pixel_array
        img= resize(img,(self.image_size, self.image_size))
        img = img.reshape((self.image_size, self.image_size, 1))
        np.stack([img, img, img], axis=2).reshape((self.image_size, self.image_size, 3))
        return img
        
    def __getitem__(self, index):
        # select batch
        indicies = list(range(index*self.batch_size, min((index*self.batch_size)+self.batch_size ,(self.df.shape[0]))))
        
        images = []
        for img_path in self.df['filename'].iloc[indicies].tolist():
            img_path = img_path+".dcm"
            img = self.load_img(os.path.join(self.images_path,img_path))
            images.append(img)
        y = self.df[['negative_exam_for_pe', 'rv_lv_ratio_gte_1', 'rv_lv_ratio_lt_1',
                     'leftsided_pe', 'chronic_pe', 'rightsided_pe',
                     'acute_and_chronic_pe', 'central_pe', 'indeterminate']].iloc[indicies].values
        return np.array(images), np.array(y)
         
    def on_epoch_end(self):
        if self.shuffle:
            self.df=self.df.sample(frac=1)
        
    def __len__(self):
        return self.nb_iteration
images_path="../input/rsna-str-pulmonary-embolism-detection/train/"
df_train= train.iloc[:20000]
df_val= train.iloc[20000:25000]
train_dataloader =  generator(df_train,images_path)
val_dataloader =  generator(df_val,images_path)
x,y = next(enumerate(train_dataloader))[1]
x.shape
inputs = Input((256, 256, 1))
Densenet_model = tf.keras.applications.DenseNet121(
            include_top=False,
            weights=None,
            input_shape=(256,256,1))

outputs = Densenet_model(inputs)
outputs = GlobalAveragePooling2D()(outputs)
outputs = Dropout(0.25)(outputs)
outputs = Dense(1024, activation='relu')(outputs)
outputs = Dropout(0.25)(outputs)
outputs = Dense(256, activation='relu')(outputs)
outputs = Dropout(0.25)(outputs)
outputs = Dense(64, activation='relu')(outputs)
nepe = Dense(1, activation='sigmoid', name='negative_exam_for_pe')(outputs)
rlrg1 = Dense(1, activation='sigmoid', name='rv_lv_ratio_gte_1')(outputs)
rlrl1 = Dense(1, activation='sigmoid', name='rv_lv_ratio_lt_1')(outputs) 
lspe = Dense(1, activation='sigmoid', name='leftsided_pe')(outputs)
cpe = Dense(1, activation='sigmoid', name='chronic_pe')(outputs)
rspe = Dense(1, activation='sigmoid', name='rightsided_pe')(outputs)
aacpe = Dense(1, activation='sigmoid', name='acute_and_chronic_pe')(outputs)
cnpe = Dense(1, activation='sigmoid', name='central_pe')(outputs)
indt = Dense(1, activation='sigmoid', name='indeterminate')(outputs)

model = Model(inputs=inputs, outputs={'negative_exam_for_pe':nepe,
                                      'rv_lv_ratio_gte_1':rlrg1,
                                      'rv_lv_ratio_lt_1':rlrl1,
                                      'leftsided_pe':lspe,
                                      'chronic_pe':cpe,
                                      'rightsided_pe':rspe,
                                      'acute_and_chronic_pe':aacpe,
                                      'central_pe':cnpe,
                                      'indeterminate':indt})


model.compile(optimizer=Adam(lr=1e-3),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()
from tensorflow.keras.utils import plot_model
plot_model(model)
hist = model.fit_generator( train_dataloader,validation_data = val_dataloader,epochs = 5)