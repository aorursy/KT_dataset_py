# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../"))

print(os.listdir("../working/"))

print(os.listdir("../lib/"))

print(os.listdir("../config/"))

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Creating the necessary directories

# !mkdir ../images/

# !mkdir ../images/train

# !mkdir ../images/val
# !pip install awscli
# !aws s3 --no-sign-request cp s3://open-images-dataset/tar/train_f.tar.gz '../images/train' # [target_dir] (46G)

# !tar -xzf ../images/train/train_f.tar.gz # unzip the files

# !rm ../images/train/train_f.tar.gz # remove the zipper file from disk
import bq_helper

from bq_helper import BigQueryHelper

# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package

# https://www.kaggle.com/paultimothymooney/how-to-query-the-open-images-dataset

open_images = bq_helper.BigQueryHelper(active_project="bigquery-public-data",

                                   dataset_name="open_images")

bq_assistant = BigQueryHelper("bigquery-public-data", "open_images")

print(bq_assistant.list_tables())
bq_assistant.head("annotations_bbox", num_rows=2)
bq_assistant.head("images", num_rows=2)
bq_assistant.head("dict", num_rows=20000)
bq_assistant.head("labels", num_rows=2)
bq_assistant.table_schema("annotations_bbox")
import urllib, urllib.request

import matplotlib.pyplot as plt

from PIL import Image



!mkdir ../input/images/



df = bq_assistant.head("images", num_rows=10)

for i in range(2):

    filename = df.iloc[i]['original_url'].split('/')[-1]

    urllib.request.urlretrieve(df.iloc[i]['original_url'],filename=os.path.join('../input/images/', filename))

    plt.figure()

    image = plt.imread(os.path.join('../input/images/', filename))

    plt.imshow(image)