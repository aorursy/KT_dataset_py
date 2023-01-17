import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from PIL import Image

import io, bson, multiprocessing

from skimage.data import imread
categories = pd.read_csv('../input/category_names.csv', index_col='category_id')
print(categories.describe(), "\n\n", "==================================================================")

print(categories.head())
# read bson file into pandas DataFrame

with open('../input/train_example.bson','rb') as b:

    df = pd.DataFrame(bson.decode_all(b.read()))

    

# convert binary image to raw image and store in the imgs column

df['imgs'] = df['imgs'].apply(lambda rec: rec[0]['picture'])

df['imgs'] = df['imgs'].apply(lambda img: Image.open(io.BytesIO(img)))
print(df.head())

print(df.imgs.head())
for i in range(16):

    plt.imshow(df.iloc[i,2])

    plt.show()