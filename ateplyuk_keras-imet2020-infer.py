import numpy as np

import pandas as pd

import os

import sys



import tensorflow as tf, tensorflow.keras.backend as K

from tensorflow.keras.models import load_model

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from matplotlib import pyplot as plt



sys.path.insert(0, '/kaggle/input/efficientnet-keras-source-code/')

import efficientnet.tfkeras as efn



print(tf.__version__)

print(tf.keras.__version__)
train_df = pd.read_csv("../input/imet-2020-fgvc7/train.csv")

train_df["attribute_ids"]=train_df["attribute_ids"].apply(lambda x:x.split(" "))

train_df["id"]=train_df["id"].apply(lambda x:x+".png")



print(train_df.shape)

train_df.head()
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()



train_df_d = pd.DataFrame(mlb.fit_transform(train_df["attribute_ids"]),columns=mlb.classes_, index=train_df.index)



print(train_df_d.shape)

train_df_d.head()
train_df_d[:1][['448','2429','782']]
label_names = train_df_d.columns
import gc



del train_df_d

gc.collect()
sam_sub_df = pd.read_csv('../input/imet-2020-fgvc7/sample_submission.csv')



sam_sub_df["id"]=sam_sub_df["id"].apply(lambda x:x+".png")



print(sam_sub_df.shape)

sam_sub_df.head()
img_size = 32
model = load_model('/kaggle/input/keras-imet2020-tpu-train/model.h5')
%%time

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_dataframe(  

        dataframe=sam_sub_df,

        directory = "../input/imet-2020-fgvc7/test",    

        x_col="id",

        target_size = (img_size,img_size),

        batch_size = 1,

        shuffle = False,

        class_mode = None

        )
%%time



test_generator.reset()

probs = model.predict_generator(test_generator, steps = len(test_generator.filenames))
probs.shape
probs[0].mean()
threshold = probs[0].mean()

labels_01 = (probs > threshold).astype(np.int)

labels_01
labels_01.shape
sub = pd.DataFrame(labels_01, columns = label_names)



print(sub.shape)

sub.head()
%%time



sub['attribute_ids']=''

for col_name in sub.columns:

    sub.ix[sub[col_name]==1,'attribute_ids']= sub['attribute_ids']+' '+col_name
sub.head()
sam_sub_df['id'] = sam_sub_df['id'].str[:-4]

sam_sub_df.head()
sam_sub_df['attribute_ids'] = sub['attribute_ids']

sam_sub_df.head()
sam_sub_df.tail()
sam_sub_df.to_csv("submission.csv",index=False)