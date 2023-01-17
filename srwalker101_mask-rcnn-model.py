# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

import tensorflow as tf

import cv2

import matplotlib.pyplot as plt



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/severstal-steel-defect-detection/train.csv")

df["image_id"] = df["ImageId_ClassId"].apply(lambda val: val.split("_")[0])

df["class_id"] = df["ImageId_ClassId"].apply(lambda val: val.split("_")[1])

df = df.rename(columns={"EncodedPixels": "encoded_pixels"})

df = df[["image_id", "class_id", "encoded_pixels"]]



# compute test/train split

vals = np.random.uniform(0, 1, len(df))

train_idx = vals < 0.8

val_idx = vals >= 0.8



df["train"] = False

df["train"][train_idx] = True



df.head()
with_pixels = df.dropna()

with_pixels.head()
filename, class_id = with_pixels.iloc[0].image_id, with_pixels.iloc[0].class_id

print(filename, class_id)
def compute_mask(row, shape):

    width, height = shape

    

    mask = np.zeros(width * height, dtype=np.uint8)

    pixels = np.array(list(map(int, row.encoded_pixels.split())))

    mask_start = pixels[0::2]

    mask_length = pixels[1::2]

        

    for s, l in zip(mask_start, mask_length):

        mask[s:s + l] = 255

        

    mask = np.flipud(np.rot90(mask.reshape((height, width))))

    return mask
def mask_to_image(mask):

    return np.transpose(np.array([mask, mask, mask]), [1, 2, 0])
!ls ../input/severstal-steel-defect-detection/

def show_image(axis, filename, df, colours):

    row_ids = np.where(df["image_id"] == filename)[0]

    if not row_ids.size:

        raise ValueError(f"Cannot find image {filename}")

        

    assert len(row_ids) <= len(colours)

    

    

    combined_image = None

    for i, (row_id, colour) in enumerate(zip(row_ids, colours)):

        row = df.iloc[row_id]

        

        filename = os.path.join("..", "input", "severstal-steel-defect-detection", "train_images", row.image_id)

        assert os.path.isfile(filename)



        data = cv2.imread(filename)

        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)

        if i == 0:

            combined_image = data



        if not isinstance(row.encoded_pixels, str):

            continue    



        width, height, _ = data.shape

        

        mask = compute_mask(row, (width, height))

        

        full_mask = np.array([

            mask * colour[0],

            mask * colour[1],

            mask * colour[2],

        ])

        mask = np.transpose(full_mask, [1, 2, 0]).astype(np.uint8)

        

        combined_image = cv2.addWeighted(mask, 0.3, combined_image, 0.7, 0)

    

    axis.imshow(combined_image)

    

test_filename = with_pixels.iloc[5].image_id

colours = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 1.0)]



fig, axis = plt.subplots(figsize=(22, 8))

show_image(axis, test_filename, df, colours)
!pip install git+https://github.com/mindriot101/Mask_RCNN
from mrcnn.utils import Dataset

from mrcnn.config import Config

from mrcnn.model import MaskRCNN
class MyConfig(Config):

#     BACKBONE = "resnet50"

    NAME = "steel"

    IMAGES_PER_GPU = 1

    GPU_COUNT = 1

    NUM_CLASSES = 1 + 4

    STEPS_PER_EPOCH = 250

    VALIDATION_STEPS = 10

    

modelconfig = MyConfig()

modelconfig.display()
class MyDataset(Dataset):

    

    SHAPE = (1600, 256)

    

    def load_from(self, df):

        self.df = df

        

        self.add_class("", 1, "class 1")

        self.add_class("", 2, "class 2")       

        self.add_class("", 3, "class 3")

        self.add_class("", 4, "class 4")

        

        for image_id, g in df.groupby("image_id"):

            filename = os.path.join("..", "input", "severstal-steel-defect-detection", "train_images", image_id)

            assert os.path.isfile(filename)

            self.add_image("", image_id, filename)

            

    def load_mask(self, image_idx):

        width, height = self.SHAPE

        

        image_id = self.image_info[image_idx]["id"]

        

        selection = self.df.query("image_id == @image_id") 

        assert len(selection)

    

        total_mask = np.zeros((height, width, 4))

        class_ids = []

        for i, (_, row) in enumerate(selection.iterrows()):

            if not isinstance(row.encoded_pixels, str):

                continue



            class_ids.append(int(row.class_id))



            mask = np.zeros(width * height, dtype=np.uint8)

            pixels = np.array(list(map(int, row.encoded_pixels.split())))

            mask_start = pixels[0::2]

            mask_length = pixels[1::2]



            for s, l in zip(mask_start, mask_length):

                mask[s:s + l] = 255



            mask = np.flipud(np.rot90(mask.reshape((width, height))))

            total_mask[:, :, i] = mask

            

        return total_mask, np.array([1, 2, 3, 4])
dataset_train = MyDataset()

dataset_train.load_from(df[df.train == True])

dataset_train.prepare()



dataset_val = MyDataset()

dataset_val.load_from(df[df.train == False])

dataset_val.prepare()
def investigate_mask(idx):

    mask, class_ids = dataset_train.load_mask(idx)

    if not mask.any():

        print("No regions found")

        return

    

    fig, axes = plt.subplots(len(class_ids), 1, figsize=(22, 8))

    try:

        axes = axes.ravel()

    except AttributeError:

        axes = [axes]

    for i, (c, ax) in enumerate(zip(class_ids, axes)):

        m = mask[:, :, i]

        ax.imshow(m, cmap="gray")

        

    

investigate_mask(0)
!curl -LO https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5
# Load the model

config = tf.ConfigProto()

config.gpu_options.allow_growth = True

session = tf.Session(config=config)



session.run(tf.global_variables_initializer())

session.run(tf.local_variables_initializer())

model = MaskRCNN(mode="training", config=modelconfig, model_dir="modeldir")

model.load_weights("mask_rcnn_coco.h5", by_name=True, exclude=["mrcnn_bbox_fc", "mrcnn_class_logits", "mrcnn_mask", "mrcnn_bbox"])
%time model.train(dataset_train, dataset_val, epochs=40, layers="heads", learning_rate=modelconfig.LEARNING_RATE)