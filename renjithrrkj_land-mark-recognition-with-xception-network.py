# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

 #   for filename in filenames:

  #      print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px

import plotly.figure_factory as ff

import plotly.graph_objects as go

from scipy import stats

import cv2

import glob

from keras.preprocessing.image import ImageDataGenerator

from keras.applications import MobileNetV2

from keras.utils import to_categorical

from keras.layers import Dense

from keras import Model

from keras.callbacks import ModelCheckpoint

from keras.models import load_model

from tensorflow.keras.applications.xception import Xception

from keras.layers import MaxPooling2D

from keras.layers import GlobalAveragePooling2D

train_df=pd.read_csv('../input/landmark-recognition-2020/train.csv')
train_df.head()
plt.figure(figsize=[15,7])

sns.distplot(train_df['landmark_id'])

plt.xlabel('landmark_id')

plt.title('distribution of lanmark')
landmark_count=pd.value_counts(train_df["landmark_id"])

landmark_count=landmark_count.reset_index()

landmark_count.rename(columns={"index":'landmark_ids','landmark_id':'count'},inplace=True)

landmark_count
plt.figure(figsize=[15,7])

sns.distplot(landmark_count)

plt.xlabel('landmark_id')

plt.title('distribution of lanmark count')

landmark_count1=landmark_count.copy()


sample = landmark_count[0:50]

sample.rename(columns={"index":'landmark_ids','landmark_id':'count'},inplace=True)

sample.sort_values(by=['count'],ascending=False,inplace=True)

sample['landmark_ids']=sample['landmark_ids'].map(str)

sample.info()

sample
plt.figure(figsize=[15,7])

ax=sns.barplot(x='landmark_ids',y='count',data=sample,order=sample['landmark_ids'],palette=sns.cubehelix_palette(50, start=9, rot=0, dark=0, light=.95, reverse=True))

for item in ax.get_xticklabels(): item.set_rotation(90)



plt.xlabel('landmark_id')

plt.ylabel('count of images')

plt.title("Count of images per landmark_id")

plt.show()
landmark_count1=landmark_count1.sort_values(by=['count'],ascending=False)

fig=px.line(landmark_count1,y='count',hover_name="landmark_ids",title="Number of images per class line")

fig.update_layout(yaxis_type="log")

fig.show()
fig=px.scatter(landmark_count1,x='landmark_ids',y='count',title="Number of images per class scatter")

fig.show()
fig=px.scatter(landmark_count1[70:],x='landmark_ids',y='count',title="Number of images per class scatter below 500")

fig.show()
sample=landmark_count1.loc[landmark_count1['count']<150]
fig=px.scatter(sample,x='landmark_ids',y='count',title="Number of images per class scatter below 150")

fig.show()
sample=landmark_count1.loc[landmark_count1['count']<50]

fig=px.scatter(sample,x='landmark_ids',y='count',title="Number of images per class scatter below 50")

fig.show()
landmark_count1.loc[landmark_count1['count']<=10000,'landmark_ids']="below 10000 and above 500 images"

landmark_count1.loc[landmark_count1['count']<=500,'landmark_ids']="below 500 and above 150 images" 

landmark_count1.loc[landmark_count1['count']<=150,'landmark_ids']="below 150 and above 50 images"

landmark_count1.loc[landmark_count1['count']<=50,'landmark_ids']="below 50 images"





landmark_count1
fig=px.pie(landmark_count1,values='count',names='landmark_ids',title='Percentage of landmarks in classes')

fig.show()
train_list = glob.glob('../input/landmark-recognition-2020/train/*/*/*/*')
example = cv2.imread(train_list[1])
plt.imshow(example)
sub = pd.read_csv("/kaggle/input/landmark-recognition-2020/sample_submission.csv")

sub["filename"] = sub.id.str[0]+"/"+sub.id.str[1]+"/"+sub.id.str[2]+"/"+sub.id+".jpg"

sub
train_df["filename"] = train_df.id.str[0]+"/"+train_df.id.str[1]+"/"+train_df.id.str[2]+"/"+train_df.id+".jpg"

train_df["label"] = train_df.landmark_id.astype(str)
from collections import Counter



c = train_df.landmark_id.values

count = Counter(c).most_common(1000)

print(len(count), count[-1])
# only keep 1000 classes

keep_labels = [i[0] for i in count]

train_keep = train_df[train_df.landmark_id.isin(keep_labels)]
val_rate = 0.2

batch_size = 32


gen = ImageDataGenerator(validation_split=val_rate)



train_gen = gen.flow_from_dataframe(

    train_keep,

    directory="/kaggle/input/landmark-recognition-2020/train/",

    x_col="filename",

    y_col="label",

    weight_col=None,

    target_size=(299, 299),

    color_mode="rgb",

    classes=None,

    class_mode="categorical",

    batch_size=batch_size,

    shuffle=True,

    subset="training",

    interpolation="nearest",

    validate_filenames=False)

    

val_gen = gen.flow_from_dataframe(

    train_keep,

    directory="/kaggle/input/landmark-recognition-2020/train/",

    x_col="filename",

    y_col="label",

    weight_col=None,

    target_size=(299, 299),

    color_mode="rgb",

    classes=None,

    class_mode="categorical",

    batch_size=batch_size,

    shuffle=True,

    subset="validation",

    interpolation="nearest",

    validate_filenames=False)
weights_xce='../input/keras-pretrained-models/xception_weights_tf_dim_ordering_tf_kernels.h5'

model  = Xception(weights=weights_xce)



model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"])
# training parameters

epochs = 4 # maximum number of epochs

train_steps = int(len(train_keep)*(1-val_rate))//batch_size

val_steps = int(len(train_keep)*val_rate)//batch_size
model_checkpoint = ModelCheckpoint("best_model.h5", save_best_only=True, verbose=1)


history = model.fit_generator(train_gen, steps_per_epoch=train_steps, epochs=epochs,validation_data=val_gen, validation_steps=val_steps, callbacks=[model_checkpoint])



model.save("model.h5")
from keras.models import load_model

best_model = load_model("model.h5")
test_gen = ImageDataGenerator().flow_from_dataframe(

    sub,

    directory="/kaggle/input/landmark-recognition-2020/test/",

    x_col="filename",

    y_col=None,

    weight_col=None,

    target_size=(299, 299),

    color_mode="rgb",

    classes=None,

    class_mode=None,

    batch_size=1,

    shuffle=True,

    subset=None,

    interpolation="nearest",

    validate_filenames=False)
print("Predicting on  available data   ")

y_pred_one_hot = best_model.predict_generator(test_gen, verbose=1, steps=len(sub))
y_pred = np.argmax(y_pred_one_hot, axis=-1)

y_prob = np.max(y_pred_one_hot, axis=-1)

print(y_pred.shape, y_prob.shape)
y_uniq = np.unique(train_keep.landmark_id.values)

print(y_uniq)

y_pred = [y_uniq[Y] for Y in y_pred]
for i in range(len(sub)):

    sub.loc[i, "landmarks"] = str(y_pred[i])+" "+str(y_prob[i])

sub = sub.drop(columns="filename")

sub.to_csv("submission.csv", index=False)

sub