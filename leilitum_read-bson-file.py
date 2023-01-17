# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import io

import os

import bson

import matplotlib.pyplot as plt

from skimage.data import imread 

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data_dir='../input'

data = bson.decode_file_iter(open(os.path.join(data_dir,'train_example.bson'),'rb'))
product_to_category=dict()

for c,d in enumerate(data):

    product_id = d['_id']

    category_id = d['category_id']

    product_to_category[product_id] = category_id

    for e,pic in enumerate(d['imgs']):

        picture = imread(io.BytesIO(pic['picture']))
product_to_category = pd.DataFrame.from_dict(product_to_category,orient='index')
product_to_category.index.name='_id'

product_to_category.rename(columns={0:'category_id'},inplace=True)
product_to_category.head()
import multiprocessing as mp
NCORE =  2

manager = mp.Manager()



product_to_category = manager.dict()
def process(q, iolock, product_to_category):

    while True:

        d=q.get()

        if d is None:

            break

        product_id = d['_id']

        category_id = d['category_id']

        product_to_category[product_id] = category_id

        for e,pic in enumerate(d['imgs']):

            picture = imread(io.BytesIO(pic['picture']))

            
data = bson.decode_file_iter(open(os.path.join(data_dir,'train_example.bson'),'rb'))

q=mp.Queue(maxsize=NCORE)

iolock=mp.Lock()

pool = mp.Pool(NCORE, initializer=process, initargs=(q, iolock, product_to_category))

for c,d in enumerate(data):

    q.put(d)

    

for _ in range(NCORE):

    q.put(None)

pool.close()

pool.join()
product_to_category=dict(product_to_category)
product_to_category