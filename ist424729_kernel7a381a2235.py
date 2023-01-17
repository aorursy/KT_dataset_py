# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.





train_data_dir = "../input/fruits-360_dataset/fruits-360/Test"

test_data_dir = "../input/fruits-360_dataset/fruits-360/Training"



train_apple = [train_data_dir + "/{}".format(i) for i in os.listdir(train_data_dir)  

                                                             if 'Apple Crimson Snow' 

                                                             or 'Apple Golden 1'

                                                             or 'Apple Golden 2'

                                                             or 'Apple Golden 3'

                                                             or 'Apple Granny Smith'

                                                             or 'Apple Red 1'

                                                             or 'Apple Red 2'

                                                             or 'Apple Red 3'

                                                             or 'Apple Red Delicious'

                                                             or 'Apple Red Yellow 1'

                                                             or 'Apple Red Yellow 2' 

                                                             in i]

train_orange = [train_data_dir + "/{}".format(i) for i in os.listdir(train_data_dir) if 'Orange' in i]

train_lemon =  [train_data_dir + "/{}".format(i) for i in os.listdir(train_data_dir) if 'Lemon' 

                                                                                     or 'Lemon Meyer' in i]

train_pear = train_lemon =  [train_data_dir + "/{}".format(i) for i in os.listdir(train_data_dir) 

                                                                                    if 'Pear Abate' 

                                                                                    or 'Pear Kaiser'

                                                                                    or 'Pear Monster' 

                                                                                    or 'Pear Red' 

                                                                                    or 'Pear Williams'

                                                                                         in i]

test_set = [test_data_dir + '/{}'.format(i) for i in os.listdir(test_data_dir)]



train_set = train_orange[:] + train_apple[:] + train_lemon[:] + train_pear[:]

random.shuffle(train_set)

print("size train set", len(train_set))

print("size test set", len(test_set))

validation_set = train_set[:124]

print("size validation set", len(validation_set))