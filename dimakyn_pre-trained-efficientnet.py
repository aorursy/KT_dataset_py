!pip install /kaggle/input/download-tf2-3-0-for-offline/tensorboard-2.3.0-py3-none-any.whl



!pip install /kaggle/input/download-tf2-3-0-for-offline/tensorflow_estimator-2.3.0-py2.py3-none-any.whl



!pip install /kaggle/input/download-tf2-3-0-for-offline/tensorflow-2.3.0-cp37-cp37m-manylinux2010_x86_64.whl
# Input data files are available in the read-only "../input/" dirimport numpy as np

import os

import numpy as np

import pandas as pd 



from keras.preprocessing.image import ImageDataGenerator



from keras.models import load_model



import tensorflow as tf
tf.__version__
train_df=pd.read_csv('../input/landmark-recognition-2020/train.csv')
train_df["filename"] = train_df.id.str[0]+"/"+train_df.id.str[1]+"/"+train_df.id.str[2]+"/"+train_df.id+".jpg"

train_df["label"] = train_df.landmark_id.astype(str)



keep_labels = train_df["label"]

train_keep = train_df[train_df.landmark_id.isin(keep_labels)]
sub = pd.read_csv("/kaggle/input/landmark-recognition-2020/sample_submission.csv")

sub["filename"] = sub.id.str[0]+"/"+sub.id.str[1]+"/"+sub.id.str[2]+"/"+sub.id+".jpg"

sub
best_model = load_model("../input/landmark-tpu/model.h5")



test_gen = ImageDataGenerator().flow_from_dataframe(

    sub,

    directory="/kaggle/input/landmark-recognition-2020/test/",

    x_col="filename",

    y_col=None,

    weight_col=None,

    target_size=(256, 256),

    color_mode="rgb",

    classes=None,

    class_mode=None,

    batch_size=1,

    shuffle=True,

    subset=None,

    interpolation="nearest",

    validate_filenames=False)
y_pred_one_hot = best_model.predict_generator(test_gen, verbose=1, steps=len(sub))
y_pred = np.argmax(y_pred_one_hot, axis=-1)

y_prob = np.max(y_pred_one_hot, axis=-1)

print(y_pred.shape, y_prob.shape)
y_uniq = np.unique(train_keep.landmark_id.values)



y_pred = [y_uniq[Y] for Y in y_pred]
for i in range(len(sub)):

    sub.loc[i, "landmarks"] = str(y_pred[i])+" "+str(y_prob[i])

sub = sub.drop(columns="filename")

sub.to_csv("/kaggle/working/submission.csv", index=False)

sub