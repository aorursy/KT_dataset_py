import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import json
import csv
from pandas import DataFrame
import gc

from IPython.display import Image
from IPython.core.display import HTML

import concurrent.futures
import requests

from PIL import Image
from tqdm import tqdm_notebook as tqdm

%matplotlib inline 

import os
print(os.listdir("../input"))
with open("../input/train.json") as datafile1: #first check if it's a valid json file or not
    train_data = json.load(datafile1)
with open("../input/test.json") as datafile2: #first check if it's a valid json file or not
    test_data = json.load(datafile2)
with open("../input/validation.json") as datafile3: #first check if it's a valid json file or not
    valid_data = json.load(datafile3)
print("####" * 10)
print("## Training Data.")
print(train_data.keys())
print(len(train_data["images"]))
print(len(train_data["annotations"]))
print(train_data["images"][:10])
print(train_data["annotations"][:10])

print("\n\n")
print("####" * 10)
print("## Validation Data.")
print(valid_data.keys())
print(len(valid_data["images"]))
print(len(valid_data["annotations"]))
print(valid_data["images"][:10])
print(valid_data["annotations"][:10])

print("\n\n")
print("####" * 10)
print("## Test Data.")
print(test_data.keys())
print(len(test_data["images"]))
print(test_data["images"][:10])
train_imgs_df = pd.DataFrame.from_records(train_data["images"])
train_imgs_df["url"] = train_imgs_df["url"].str[0]
train_labels_df = pd.DataFrame.from_records(train_data["annotations"])
train_df = pd.merge(train_imgs_df,train_labels_df,on="image_id",how="outer")
print(train_df.head())

valid_imgs_df = pd.DataFrame.from_records(valid_data["images"])
valid_imgs_df["url"] = valid_imgs_df["url"].str[0]
valid_labels_df = pd.DataFrame.from_records(valid_data["annotations"])
valid_df = pd.merge(valid_imgs_df,valid_labels_df,on="image_id",how="outer")
print(valid_df.head())

test_df = pd.DataFrame.from_records(test_data["images"])
test_df["url"] = test_df["url"].str[0]
print(test_df.head())
del train_data
del valid_data
del test_data
gc.collect()
print("####" * 10)
print("## Training Data.")
print(train_df.isna().any())

print("\n\n")
print("####" * 10)
print("## Validation Data.")
print(valid_df.isna().any())

print("\n\n")
print("####" * 10)
print("## Testing Data.")
print(test_df.isna().any())
## Checking number of unique labels.
print(train_df["label_id"].nunique())
print(valid_df["label_id"].nunique())
#Class distribution
plt.figure(figsize = (20, 16))
plt.title('Train Category Distribuition')
sns.distplot(train_df['label_id'])

plt.show()
#Class distribution
plt.figure(figsize = (20, 16))
plt.title('Valid Category Distribuition')
sns.distplot(valid_df['label_id'])

plt.show()
def display_category(label_id, df, num_disp=8):
    samp_df = df[df["label_id"] == label_id].sample(n=num_disp)
    urls = samp_df["url"].tolist()
    img_style = "width: 180px; margin: 0px; float: left; border: 1px solid black;"
    images_list = ''.join([f"<img style='{img_style}' src='{u}' />" for u in urls])
    header_str = "<h2>Label {:d}</h2>".format(label_id)
    display(HTML(header_str))
    display(HTML(images_list))
for label in sorted(pd.unique(train_df["label_id"]).ravel().tolist()):
    display_category(label, train_df)
for label in sorted(pd.unique(valid_df["label_id"]).ravel().tolist()):
    display_category(label, valid_df,num_disp=5)
