import os

import pandas as pd

import numpy as np

from pathlib import Path

from PIL import Image

from tqdm.notebook import tqdm
data_path = Path('../input/prostate-cancer-grade-assessment/')

os.listdir(data_path)
!cd ../input/prostate-cancer-grade-assessment/; du -h
os.listdir(data_path/'train_images')
import pandas as pd

train_df = pd.read_csv(data_path/'train.csv')

train_df.head(10)
print('Number of whole-slide images in training set: ', len(train_df))
sample_image = train_df.iloc[np.random.choice(len(train_df))].image_id

print(sample_image)
import openslide
openslide_image = openslide.OpenSlide(str(data_path/'train_images'/(sample_image+'.tiff')))
openslide_image.properties
img = openslide_image.read_region(location=(0,0),level=0,size=(openslide_image.level_dimensions[0][0],openslide_image.level_dimensions[0][1]))

img
Image.fromarray(np.array(img.resize((512,512)))[:,:,:3])
for i in tqdm(train_df['image_id'],total=len(train_df)):

    openslide_image = openslide.OpenSlide(str(data_path/'train_images'/(i+'.tiff')))

    img = openslide_image.read_region(location=(0,0),level=2,size=(openslide_image.level_dimensions[2][0],openslide_image.level_dimensions[2][1]))

    Image.fromarray(np.array(img.resize((512,512)))[:,:,:3]).save(i+'.jpeg')

    