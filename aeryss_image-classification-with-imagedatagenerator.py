# Load libraries
import pandas as pd
import tensorflow as tf
# Load csv files
data_dir = "../input/vietai-c6-assignment3-extracted-dataset/train.csv"
sub_dir = "../input/vietai-c6-assignment3-extracted-dataset/sample_submission.csv"
train_df = pd.read_csv(data_dir)
submission_df = pd.read_csv(sub_dir)
train_df
submission_df
# Create one-hot encoding
classes = ["book", "can", "cardboard", "glass_bottle", "pen", "plastic_bottle"]
train_y = train_y = train_df.label
num_classes = len(np.unique(train_y))
y_ohe = tf.keras.utils.to_categorical(train_y, num_classes=num_classes)
y_ohe
# Show sample images
from PIL import Image
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np

def show_gallery(df, n=5, shuffle=True):
    plt.subplots(figsize=(20,20))
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    k=1
    for i in range(n*n):
        im = cv2.imread(os.path.join("../input/vietai-c6-assignment3-extracted-dataset/train/" + classes[df.loc[k, "label"]],df.loc[k,"image"]))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        label = classes[df.loc[k,"label"]]
        plt.subplot(n,n,k)
        plt.imshow(im)
        plt.title("Label: {}".format(label))
        plt.axis("off")
        k += 1

show_gallery(train_df)
# Load images using ImageDataGenerator

from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 224
BATCH_SIZE = 32

train_data_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2) 
train_gen = train_data_gen.flow_from_directory("../input/vietai-c6-assignment3-extracted-dataset/train", batch_size=BATCH_SIZE,
                                              target_size=(IMG_SIZE, IMG_SIZE), subset="training")
valid_gen = train_data_gen.flow_from_directory("../input/vietai-c6-assignment3-extracted-dataset/train", batch_size=BATCH_SIZE,
                                              target_size=(IMG_SIZE, IMG_SIZE), subset="validation")
# Fit your data as normal

# model.fit(train_gen, validation_data = valid_gen)