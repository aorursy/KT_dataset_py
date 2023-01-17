import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image
import openslide
train_img_path = "/kaggle/input/prostate-cancer-grade-assessment/train_images"
label_path = "/kaggle/input/prostate-cancer-grade-assessment/train_label_masks"

train_img = [img for img in os.listdir(train_img_path)]
train_label = [label for label in os.listdir(label_path)]

train_img = list(sorted(train_img))
train_label = list(sorted(train_label))

SIZE = 200

def preprocessing_img(img):
    slide = openslide.OpenSlide(img)
    img = np.array(slide.get_thumbnail(size=(SIZE, SIZE)))
    img = Image.fromarray(img)
    img = img.resize((SIZE, SIZE))
    img = np.array(img)
    return img

img_array = []
for i in range(10):
    img = train_img_path + "/" + train_img[i]
    img = preprocessing_img(img)
    img_array.append(img)
plt.imshow(img_array[0])
