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

import bson

import os

import collections

from tqdm import tqdm_notebook
train_folder      = '../output/train'

validation_folder = '../output/validation'



# Create train folder

if not os.path.exists(train_folder):

    os.makedirs(train_folder)

    

# Create validation folder

if not os.path.exists(validation_folder):

    os.makedirs(validation_folder)

    
# Create categories folders

categories = pd.read_csv('../input/category_names.csv', index_col='category_id')



for category in tqdm_notebook(categories.index):

    os.mkdir(os.path.join(train_folder, str(category)))

    os.mkdir(os.path.join(validation_folder, str(category)))

    
num_products = 7069896  # 7069896 for train and 1768182 for test

num_prod_train = num_products*0.8   #set 80% of the data as the training set. Leave the remainder as validation set

print('training set will have ', num_prod_train, 'items')



bar = tqdm_notebook(total=num_products)

counter = 0

with open('../input/train.bson', 'rb') as fbson:



    data = bson.decode_file_iter(fbson)

    

    for c, d in enumerate(data):

        category = d['category_id']

        _id = d['_id']

        counter += 1



        for e, pic in enumerate(d['imgs']):

            if counter < num_prod_train :

                fname = os.path.join(train_folder, str(category), '{}-{}.jpg'.format(_id, e))                

            else:

                fname = os.path.join(validation_folder, str(category), '{}-{}.jpg'.format(_id, e))

            with open(fname, 'wb') as f:

                f.write(pic['picture'])



        bar.update()