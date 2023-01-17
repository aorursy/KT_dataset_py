from bs4 import BeautifulSoup
import random
import os, csv
import numpy as np
import tensorflow
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
seed1=193
tensorflow.random.set_seed(seed1)
np.random.seed(seed1)
random.seed(seed1)
import cv2
a=cv2.imread("../input/simple-object-detection/datasets/images/a (101).jpg",0)
print(a)
datasets_directory = "../input/simple-object-detection/datasets"
annotations_directory="../input/simple-object-detection/datasets/annotations/"
format='.jpg'
N = []      
for r, d, f in os.walk(datasets_directory, topdown=False):
    N.append(f)   
K=np.arange(len(N[0]))
random.shuffle(K)
for i in K:  
    annotation_file=annotations_directory+N[0][i]
    ds = BeautifulSoup(open(annotation_file).read(), "html.parser")
    # Iterating each object elements
    for o in ds.find_all("object"):
        class_label = o.find("name").string
        x_min = max(0, int(float(o.find("xmin").string)))
        y_min = max(0, int(float(o.find("ymin").string)))
        x_max = min(int(ds.find("width").string), int(float(o.find("xmax").string)))
        y_max = min(int(ds.find("height").string), int(float(o.find("ymax").string)))
        # controlling errors
        if x_min >= x_max or y_min >= y_max:
            continue
        elif x_max <= x_min or y_max <= y_min:
            continue
        line = [N[1][i], str(x_min), str(y_min), str(x_max), str(y_max), str(class_label)]
        with open("datasets.csv", 'a', newline='') as f:
                csv.writer(f).writerow(line)
print("datasets.csv has been created...")
df = pd.read_csv('datasets.csv',header = None,names=["image_tag", "left", "top", "right","bottom",'a'])
df=df.drop(['a'], axis=1)
df.head()
# normalise locations (output coordinates)
df["left"]=df["left"]/224
df["top"]=df["top"]/224
df["right"]=df["right"]/224
df["bottom"]=df["bottom"]/224
df.head()
# train and test split
rt=0.2
ix=int((1-rt)*len(df))
df1 = df.iloc[:ix,:] 
df2 = df.iloc[ix+1:,:]
datagen = ImageDataGenerator(rescale=1./255)
train_g = datagen.flow_from_dataframe(
    df1, directory='/kaggle/input/datasets/images/',
    x_col="image_tag",y_col=["left", "top", "right","bottom"],
    target_size=(224, 224),batch_size=5, 
    class_mode="raw",subset="training")
valid_g = datagen.flow_from_dataframe(
    df2, directory='/kaggle/input/datasets/images/',
    x_col="image_tag",y_col=["left", "top", "right","bottom"],
    target_size=(224, 224),batch_size=5, 
    class_mode="raw",subset="training")
# Model Setup
model = tf.keras.models.Sequential()
model.add(tf.keras.applications.InceptionV3(weights="imagenet", include_top=False, input_shape=(224, 224, 3)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation="relu"))
model.add(tf.keras.layers.Dense(4, activation="relu"))
model.summary()
model.compile(tf.keras.optimizers.SGD(learning_rate=0.1),loss='categorical_crossentropy',metrics=['accuracy'])
# Training
model.fit(train_g, steps_per_epoch=17, validation_data=valid_g, validation_steps=4, epochs=20)