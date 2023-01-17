# Imports

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import io

import bson                       # this is installed with the pymongo package

import matplotlib.pyplot as plt

from skimage.data import imread   # or, whatever image library you prefer

import multiprocessing as mp      # will come in handy due to the size of the data
cat_names_df = pd.read_csv("../input/category_names.csv")

samp_sub_df = pd.read_csv("../input/sample_submission.csv")
print(cat_names_df.head())

print(samp_sub_df.head())
print(len(cat_names_df))
print(len(samp_sub_df))
samp_sub_df["category_id"] = np.random.choice(cat_names_df["category_id"].values,len(samp_sub_df))
print(samp_sub_df.head())
#samp_sub_df.to_csv("rand_submission.csv.gz", compression="gzip", index=False)

samp_sub_df.to_csv("rand_submission.csv", index=False)
data = bson.decode_file_iter(open('../input/train_example.bson', 'rb'))



for c, d in enumerate(data):

    product_id = d['_id']

    category_id = d['category_id'] # This won't be in Test data

    

    for e, pic in enumerate(d['imgs']):

        pic = imread(io.BytesIO(pic['picture']))

        plt.imshow(pic);

        plt.show()

        # do something with the picture, etc