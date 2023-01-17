# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import os



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input/train"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
fish_labels = os.listdir('../input/train')

fish_labels.remove('.DS_Store')

print(fish_labels)



fish_dict = {} # Maps fish labels to images

for fish in fish_labels:

    fish_dict[fish] = tf.train.string_input_producer(os.listdir('../input/train/'+fish)) 

    # string_input_producer : Output strings (e.g. filenames) to a queue for an input pipeline.
