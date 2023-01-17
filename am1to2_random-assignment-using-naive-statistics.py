import numpy as np

import pandas as pd

import io

import bson

import matplotlib.pyplot as plt

from skimage.data import imread

from tqdm import tqdm_notebook
categories = pd.read_csv('../input/category_names.csv', index_col='category_id')
prod_id = []

prod_category = []

prod_num_imgs = []



num_dicts = 7069896 # according to data page



# This will take about 02m15s to complete

with open('../input/train.bson', 'rb') as f, tqdm_notebook(total=num_dicts) as bar:

        

    data = bson.decode_file_iter(f)



    for c, d in enumerate(data):

        bar.update()

        prod_id.append(d['_id'])

        prod_category.append(d['category_id'])

        prod_num_imgs.append(len(d['imgs']))
df_dict = {

    'category': prod_category,

    'num_imgs': prod_num_imgs

}

df = pd.DataFrame(df_dict, index=prod_id)

del df_dict # Free memory
df.num_imgs.value_counts().plot(kind='bar');

print("## Total number of images: {:d}".format(df.num_imgs.sum()))

print("## Total number of categories: {:d}".format(len(pd.unique(df.category))))
cat_counts = df.category.value_counts().to_frame()

cat_counts = cat_counts / cat_counts["category"].sum()

print(cat_counts.head())

print(pd.unique(df.category))
cat_counts.sort_values(by="category",inplace=True)

bot_5_categories = cat_counts.head()

top_5_categories = cat_counts.tail()

print(bot_5_categories)

print(top_5_categories)
samp_sub_df = pd.read_csv("../input/sample_submission.csv")

samp_sub_df["category_id"] = np.random.choice(cat_counts.index,size=len(samp_sub_df),p=cat_counts["category"].values)

samp_sub_df.to_csv("baised_rand_submission.csv", index=False)