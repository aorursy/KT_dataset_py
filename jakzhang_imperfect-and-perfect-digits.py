# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os



# Any results you write to the current directory are saved as output.
# This tells matplotlib not to try opening a new window for each plot.

%matplotlib inline



# Import a bunch of libraries.

import time

import os



import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import BernoulliNB

from sklearn.naive_bayes import GaussianNB

from numpy import math



# Set the randomizer seed so results are the same each time.

np.random.seed(0)



# Prepare Dataset

# load data

train = pd.read_csv(r"../input/train.csv",dtype = np.float32)



# split data into features(pixels) and labels(numbers from 0 to 9)

targets_numpy = train.label.values

features_numpy = train.loc[:,train.columns != "label"].values/255 # normalization



# train test split. Size of train data is 80% and size of test data is 20%. 

train_data, test_data, train_labels, test_labels = train_test_split(features_numpy,

                                                                             targets_numpy,

                                                                             test_size = 0.2,

                                                                             random_state = 42)
def imperfect_numbers(num_examples=10):

    

    #Look for examples of each digit

    from itertools import chain



    zero = list(np.where(train_labels==0)[0][:num_examples])

    one = list(np.where(train_labels==1)[0][:num_examples])

    two = list(np.where(train_labels==2)[0][:num_examples])

    three = list(np.where(train_labels==3)[0][:num_examples])

    four = list(np.where(train_labels==4)[0][:num_examples])

    five = list(np.where(train_labels==5)[0][:num_examples])

    six = list(np.where(train_labels==6)[0][:num_examples])

    seven = list(np.where(train_labels==7)[0][:num_examples])

    eight = list(np.where(train_labels==8)[0][:num_examples])

    nine = list(np.where(train_labels==9)[0][:num_examples])

    

    #Store them together in one list and reshape them

    all=list(chain(zero, one, two, three, four, five, six, seven, eight, nine))

    xgrid=train_data[all]

    xgrid=list(xgrid)

    xgrid=[x.reshape(28,28) for x in xgrid]

    

    #Create a grid subplot 

    plt.figure(figsize=(28,28))

    for i in range(0,10*num_examples):

        plt.subplot(10,num_examples,i+1)

        plt.imshow(xgrid[i],cmap=plt.get_cmap('gray_r'))

        plt.tick_params(left='off', bottom='off', right='off', top='off', labelleft='off', labelbottom='off')



imperfect_numbers(10)
def perfect_numbers():

    

    naive_binom = BernoulliNB(binarize = 0.5)

    naive_binom.fit(train_data,train_labels)

    prob = (np.exp(naive_binom.feature_log_prob_))

    

    generate = list()

    for i in range(1,101):

        generate.append((prob[math.ceil((i/10)-1)])*(np.random.rand(784,)))

        i=i+1

    generate = [x.reshape(28,28) for x in generate]

    

    plt.figure(figsize=(28,28))

    for i in range(0,100):

        plt.subplot(10,10,i+1)

        plt.imshow(generate[i],cmap=plt.get_cmap('gray_r'))

        plt.tick_params(left='off', bottom='off', right='off', top='off', labelleft='off', labelbottom='off')



perfect_numbers()