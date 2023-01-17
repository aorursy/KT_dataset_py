# Useful Libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy as sp

import sklearn

import random

import time 



#Data Modelling Libraries

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns

import IPython

from IPython import display #for prettier DataFrames



#Visualization

%matplotlib inline

mpl.style.use('ggplot')

sns.set_style('white')

pylab.rcParams['figure.figsize'] = 12,8



# Input data files are available in the "../input/" directory.

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data_raw = pd.read_csv('../input/train.csv')

#We break our dataset into 3 splits : train, test, and validation. 

    #The test file provided is the validation file that we will use for submission.

data_val = pd.read_csv('../input/test.csv')

    #We copy the train set so that we can make changes without worrying about it. 

        #Reminder if I want to implement data_raw.copy(deep = True) : Deep copy creates new id's of every object it contains while 

        #normal copy only copies the elements from the parent and creates a new id for a variable to which it is copied to.

data1 = data_raw.copy(deep = True)

    #We pass data1 & data_val by reference to clean them both at once

data_cleaner = [data1, data_val]
#preview data : 

print ('Info on our train set\n')

print (data_raw.info())



print ('What it looks like at the top\n')

data_raw.head()
print ('What it looks like at the bottom\n')

data_raw.tail()
data_raw.sample(10)

#There is going to be some cleaning to do in the way variables are coded
print ('Train columns with null values:\n', data1.isnull().sum())

print ('-' * 25)

print ('Test columns with null values:\n', data_val.isnull().sum())

print ('-' * 25)
data_raw.describe(include = 'all')
# Clean the 'belongs_to_collection' column 

# --> apply() : Apply a function along an axis of the DataFrame

data1["collection_bin"] = data1["belongs_to_collection"].apply(lambda x: 1 if not pd.isnull(x) else np.nan)



col = 'belongs_to_collection'

data_collec = data1[['id', col]]



collections = {}



for _, x in data_collec.values:

    

    value = eval(str(x))

    print(value)





def collection_one_hot_encoder(collec : str):

    """ Function to encode movies with 1

         if they belong to the same collection

         and 0 if not.

    """

    data1[collec] = data1["belongs_to_collection"].apply(lambda x: 1 if collec in str(x) else 0)

    

value = eval(str(data_collec.values[1]))