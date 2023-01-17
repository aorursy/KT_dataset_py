# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf

import tensorflow.keras as keras



print(tf.__version__)
N_CLASSES =  2000 # Number of classes to keep

BATCH_SIZE = 32
train = pd.read_csv("/kaggle/input/landmark-recognition-2020/train.csv")

train["filename"] = train.id.str[0]+"/"+train.id.str[1]+"/"+train.id.str[2]+"/"+train.id+".jpg"

train["label"] = train.landmark_id.astype(str)

train
y = train.landmark_id.values

n_classes = np.max(y)

print(n_classes)
top_classes = train['landmark_id'].value_counts()[:N_CLASSES].index.tolist()

train_keep = train[train['landmark_id'].isin(top_classes)]
base = tf.keras.applications.ResNet50(

    include_top=False,

    weights="/kaggle/input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5",

    input_tensor=None,

    input_shape=None,

    pooling=None,

    #classes=1000,

    #classifier_activation="softmax",

)

base.trainable = False



inputs = keras.Input(shape=(224, 224, 3))

x = base(inputs, training=False)

x = keras.layers.GlobalAveragePooling2D()(x)

x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout

out = tf.keras.layers.Dense(N_CLASSES, activation='softmax')(x)

model = keras.Model(inputs, out)



model.summary()
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"])
idg = tf.keras.preprocessing.image.ImageDataGenerator(

    featurewise_center=False,

    samplewise_center=False,

    featurewise_std_normalization=False,

    samplewise_std_normalization=False,

    zca_whitening=False,

    zca_epsilon=1e-06,

    rotation_range=15,

    width_shift_range=0.0,

    height_shift_range=0.0,

    brightness_range=(0.95, 1.05),

    shear_range=0.0,

    zoom_range=(0.95, 1.05),

    channel_shift_range=0.0,

    fill_mode='nearest',

    cval=0.0,

    horizontal_flip=False,

    vertical_flip=False,

    rescale=1/.255,

    preprocessing_function=None,

    data_format=None,

    validation_split=0.1,

    dtype=None

)





train_gen = idg.flow_from_dataframe(

    train_keep,

    directory="/kaggle/input/landmark-recognition-2020/train/",

    x_col="filename",

    y_col="label",

    weight_col=None,

    target_size=(224, 224),

    color_mode="rgb",

    classes=None,

    class_mode="categorical",

    batch_size=BATCH_SIZE,

    shuffle=True,

    subset="training",

    interpolation="nearest",

    validate_filenames=False)

    

val_gen = idg.flow_from_dataframe(

    train_keep,

    directory="/kaggle/input/landmark-recognition-2020/train/",

    x_col="filename",

    y_col="label",

    weight_col=None,

    target_size=(224, 224),

    color_mode="rgb",

    classes=None,

    class_mode="categorical",

    batch_size=BATCH_SIZE,

    shuffle=True,

    subset="validation",

    interpolation="nearest",

    validate_filenames=False)
# training parameters

EPOCHS = 20

train_steps = int(len(train_keep)*(1-0.1))//BATCH_SIZE

val_steps = int(len(train_keep)*0.1)//BATCH_SIZE



train_steps = train_steps * 0.2

val_steps = val_steps * 0.2
history = model.fit(train_gen, validation_data=val_gen, steps_per_epoch=train_steps, validation_steps=val_steps)
sub = pd.read_csv("/kaggle/input/landmark-recognition-2020/sample_submission.csv")

sub["filename"] = sub.id.str[0]+"/"+sub.id.str[1]+"/"+sub.id.str[2]+"/"+sub.id+".jpg"



test_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/.255).flow_from_dataframe(

    sub,

    directory="/kaggle/input/landmark-recognition-2020/test/",

    x_col="filename",

    y_col=None,

    weight_col=None,

    target_size=(224, 224),

    color_mode="rgb",

    classes=None,

    class_mode=None,

    batch_size=1,

    shuffle=True,

    subset=None,

    interpolation="nearest",

    validate_filenames=False)



y_pred = model.predict(test_gen, verbose=1)
y_pred
y_prob = np.max(y_pred, axis=-1)

y_pred = np.argmax(y_pred, axis=-1)

print(y_pred.shape, y_prob.shape)
thresh = 0.5



for i in range(len(sub)):

    prob = y_prob[i]

    pred = y_pred[i]

    if prob >= 0.3:

        st = str(pred)+" "+str(prob)

    else:

        st = ""

    sub.loc[i, "landmarks"] = st

sub = sub.drop(columns="filename")

sub.to_csv("submission.csv", index=False)

sub
prob