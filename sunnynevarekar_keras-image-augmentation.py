import os
from os.path import join
import numpy as np
import pandas as pd
import random
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from keras.applications.inception_v3 import preprocess_input
%matplotlib inline

labels = pd.read_csv('../input/labels.csv')
print(labels.shape)
labels.head()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_labels = le.fit_transform(labels['breed'])
labels['class'] = train_labels
labels.head()
#load first image fro labels.csv
train_dir = '../input/train'
ids = labels.loc[0:4, 'id']
pil_imgs = [load_img(join(train_dir, id+'.jpg'), target_size=(299, 299)) for id in ids]
imgs = [img_to_array(pil_img) for pil_img in pil_imgs]
fig = plt.figure(1, figsize=(20, 4))
grid = ImageGrid(fig, 111, nrows_ncols=(1, 5), axes_pad=0.1)

for i in range(5):
    grid[i].imshow(imgs[i].astype(np.uint8))
    grid[i].set_title(labels.loc[i, 'breed'])
#transform and show images
fig = plt.figure(1, figsize=(16, 16))
grid = ImageGrid(fig, 111, nrows_ncols=(4, 4), axes_pad=0.1)

datagen = ImageDataGenerator(rotation_range=40, zoom_range=0.2)
for i in range(4):
    for batch in datagen.flow(np.array(imgs), batch_size=4):
        for j, img in enumerate(batch):
            grid[j+i*4].imshow(img.astype(np.uint8)) 
        break
