# !pip install ipython-autotime

# %load_ext autotime



import numpy as np

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

%matplotlib inline

import plotly.express as px

import plotly.figure_factory as ff

import plotly.graph_objects as go

from scipy import stats

import cv2

import glob

from keras.preprocessing.image import ImageDataGenerator

# from keras.applications import MobileNetV2

from keras.utils import to_categorical

from keras.layers import Dense

from keras import Model

from keras.callbacks import ModelCheckpoint, EarlyStopping

from keras.models import load_model, Model

from keras.applications import MobileNetV2

from keras.optimizers import Adam

# from tensorflow.keras.applications.xception import Xception

import tensorflow as tf

import tensorflow.keras.layers as L

from collections import Counter





# import efficientnet.tfkeras as efn
train_df=pd.read_csv('../input/landmark-recognition-2020/train.csv')

print(train_df.head())
landmark_counts=pd.value_counts(train_df["landmark_id"])

landmark_counts=landmark_counts.reset_index()

landmark_counts.rename(columns={"index":'landmark_ids','landmark_id':'count'},inplace=True)

landmark_counts
train_img_name = glob.glob('../input/landmark-recognition-2020/train/*/*/*/*')

#Visualize some img

fig = plt.figure(figsize=(16,16))



for i in range(6):

    fig.add_subplot(2,3,i+1)

    img = cv2.imread(train_img_name[i+10])

    plt.imshow(img)

    

plt.show()
train_df["filename"] = train_df.id.str[0]+"/"+train_df.id.str[1]+"/"+train_df.id.str[2]+"/"+train_df.id+".jpg"

train_df["label"] = train_df.landmark_id.astype(str)

print(train_df.head(3))
n_class = len(np.unique(train_df.landmark_id.values))

print(n_class)
# Threshold for number of data each category

# threshold = 100



# landmark_count_over_threshold = landmark_counts[landmark_counts['count'] > threshold]

# length_landmark_count_over_threshold = len(landmark_count_over_threshold)

# print(length_landmark_count_over_threshold)



# landmark_id_values = train_df.landmark_id.values

# count = Counter(landmark_id_values).most_common(length_landmark_count_over_threshold)



# print(len(count))

# print(count[0])

# print(count[-1])

# keep_landmark_id = []



# for i in count:

#     keep_landmark_id.append(i[0])

    

# train_df = train_df[train_df.landmark_id.isin(keep_landmark_id)]

# print(len(train_df))

# print(train_df.head(10))





val_ratio = 0.25

batch_size = 128





gen = ImageDataGenerator(validation_split=val_ratio)



train_gen = gen.flow_from_dataframe(

    train_df,

    directory="/kaggle/input/landmark-recognition-2020/train/",

    x_col="filename",

    y_col="label",

    weight_col=None,

    target_size=(256, 256),

    color_mode="rgb",

    classes=None,

    class_mode="categorical",

    batch_size=batch_size,

    shuffle=True,

    seed = 44,

    subset="training",

    interpolation="nearest",

    validate_filenames=False)

    

val_gen = gen.flow_from_dataframe(

    train_df,

    directory="/kaggle/input/landmark-recognition-2020/train/",

    x_col="filename",

    y_col="label",

    weight_col=None,

    target_size=(256, 256),

    color_mode="rgb",

    classes=None,

    class_mode="categorical",

    batch_size=batch_size,

    shuffle=True,

    seed = 44,

    subset="validation",

    interpolation="nearest",

    validate_filenames=False)
def my_model(input_shape,nclass,dropout, learning_rate = 0.001):

    base_model = MobileNetV2(weights = None, include_top = False)

    

    model_input = L.Input(input_shape)

    x = base_model(model_input)

    x = L.GlobalAveragePooling2D()(x)

    

    y = L.Dense(512,activation='relu')(x)

    y = L.Dropout(dropout)(y)

    y = L.Dense(512,activation='relu')(y)

    y = L.Dropout(dropout)(y)

    

    y_h = L.Dense(nclass, activation = 'softmax', name = 'Id')(y)

    

    model = Model(inputs=model_input, outputs= y_h)

    

    optimizer = Adam(learning_rate=learning_rate)

    

    model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics='accuracy')

    

    return model

model = my_model(input_shape = (224,224,3), nclass = n_class, dropout = 0.4)



model.summary()
# model = load_model("../input/my-model/last_model.h5")
epochs = 1

train_steps = int(len(train_df)*(1-val_ratio))//batch_size

val_steps = int(len(train_df)*val_ratio)//batch_size



early_stopping = EarlyStopping(monitor='val_loss', mode='min',patience=6)

model_checkpoint = ModelCheckpoint("best_model.h5", monitor='loss', save_best_only=True, verbose=1)



history = model.fit_generator(train_gen, steps_per_epoch=train_steps, epochs=epochs,validation_data=val_gen, validation_steps=val_steps, callbacks=[model_checkpoint, early_stopping])

# history = model.fit_generator(train_gen, steps_per_epoch=train_steps, epochs=epochs,validation_data=val_gen, validation_steps=val_steps)

model.save("last_model.h5")
model.save("last_model.h5")
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()
my_model = load_model("last_model.h5")
test_df = pd.read_csv("/kaggle/input/landmark-recognition-2020/sample_submission.csv") #?? May be here

test_df["filename"] = test_df.id.str[0]+"/"+test_df.id.str[1]+"/"+test_df.id.str[2]+"/"+test_df.id+".jpg"

print(test_df.head(3))
test_gen = ImageDataGenerator().flow_from_dataframe(

    test_df,

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
test_steps = len(test_df)



y_pred_oh = my_model.predict_generator(test_gen, verbose=1, steps = test_steps)

print(y_pred_oh.shape)

print(y_pred_oh[:2])



y_pred = np.argmax(y_pred_oh, axis=-1)

print(y_pred.shape)

print(y_pred[:2])



y_prob = np.max(y_pred_oh, axis=-1)

print(y_prob.shape)

print(y_prob[:2])
y_landmark_id = np.unique(train_df.landmark_id.values)

y_pred_id = [y_landmark_id[Y] for Y in y_pred]

# print(y_pred)

for i in range(test_steps):

    test_df.loc[i, "landmarks"] = str(y_pred_id[i])+" "+str(y_prob[i])

test_df = test_df.drop(columns="filename")

test_df.to_csv("/kaggle/working/submission.csv", index=False)

print(test_df)