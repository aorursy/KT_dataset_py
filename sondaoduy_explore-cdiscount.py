# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import io

import bson                       # this is installed with the pymongo package

import matplotlib.pyplot as plt

from skimage.data import imread   # or, whatever image library you prefer

import multiprocessing as mp      # will come in handy due to the size of the data
data = bson.decode_file_iter(open('../input/train_example.bson', 'rb'))



prod_to_category = dict()

img_of_product = dict()



for c, d in enumerate(data):

    product_id = d['_id']

    category_id = d['category_id'] # This won't be in Test data

    if category_id in prod_to_category:

        prod_to_category[category_id].append(product_id)

    else:

        prod_to_category[category_id] = [product_id]



    #print("%s : %s " %(c,len(d['imgs'])))

    for e, pic in enumerate(d['imgs']):

        picture = imread(io.BytesIO(pic['picture']))

        if product_id in img_of_product:

            img_of_product[product_id].append(picture)

        else:

            img_of_product[product_id] = [picture]
rows, cols = 14, 8

fig, ax = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))

ax = ax.ravel()

i = 0

for key in img_of_product.keys():

    for img in img_of_product[key]:

        ax[i].imshow(img)

        i = i + 1

plt.tight_layout()

plt.show()

import scipy.io as sio

sio.savemat('product_categories.mat', {'product_categories':prod_to_category})

sio.savemat('img_by_id.mat', {'img_by_id':img_of_product})
i = 0

for key in img_of_product.keys():

    for img in img_of_product[key]:

        if i == 7:

            plt.imshow(img)

        i = i + 1