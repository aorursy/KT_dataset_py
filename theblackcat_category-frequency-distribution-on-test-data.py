# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import io

import bson     

prod_to_category = dict()
def calculate_ratio(panda_frame,n,total):

    panda_10k = panda_frame.sort_values(

    "frequency", ascending=False).head(n).reset_index()

    top_total = 0

    for data in panda_10k['frequency']:

        top_total += data

    print("Total Image {}, Ratio : {}".format(top_total,1.0*top_total/total))

    return top_total
data = bson.decode_file_iter(open('../input/train.bson', 'rb'))

category_frequency = dict()

total_product = 0

for c, d in enumerate(data):

    total_product += 1

    product_id = d['_id']

    category_id = d['category_id'] # This won't be in Test data

    if category_id not in category_frequency:

        category_frequency[category_id] = 1

    else:

        category_frequency[category_id] += 1
table_frame = []

for key,item in category_frequency.items():

    table_frame.append([key,item])

panda_frame = pd.DataFrame(table_frame, columns=['category_id','frequency'])
n = 1000

calculate_ratio(panda_frame,n,total_product)
n = 2000

calculate_ratio(panda_frame,n,total_product)
panda_10k = panda_frame.sort_values(

    "frequency", ascending=False).head(n).reset_index()

data = bson.decode_file_iter(open('../input/train.bson', 'rb'))

image_frequency = dict()

total_images = 0

for c, d in enumerate(data):

    product_id = d['_id']

    category_id = d['category_id'] # This won't be in Test data

    if category_id not in image_frequency:

        image_frequency[category_id] = 1

    for e, pic in enumerate(d['imgs']):

        total_images += 1

        image_frequency[category_id] += 1
table_frame_2 = []

for key,item in image_frequency.items():

    table_frame_2.append([key,item])

panda_frame_2 = pd.DataFrame(table_frame_2, columns=['category_id','frequency'])
n = 1000

total_images = calculate_ratio(panda_frame_2,n,total_images)
n = 2000

calculate_ratio(panda_frame_2,n,total_images)