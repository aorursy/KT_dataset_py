# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import shutil

import random

import os



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_csv = pd.read_csv('../input/rsna-str-pulmonary-embolism-detection/train.csv')

test_csv = pd.read_csv('../input/rsna-str-pulmonary-embolism-detection/test.csv')



print(train_csv.shape)

print(test_csv.shape)
train_csv.head()
test_csv.head()
tr_si_list = list(train_csv['StudyInstanceUID'].unique())

te_si_list = list(test_csv['StudyInstanceUID'].unique())



print(tr_si_list[:10])

print(te_si_list[:10])
copy_train_si = random.sample(tr_si_list, 5)

copy_test_si = random.sample(te_si_list, 2)



print(len(copy_train_si))

print(len(copy_test_si))
os.mkdir('../working/train')

os.mkdir('../working/test')
# train

i=0

for folder in copy_train_si:

    src='../input/rsna-str-pulmonary-embolism-detection/train/'+folder+'/'

    dst='../working/train/'+folder+'/'

    shutil.copytree(src, dst)

    i+=1

    print(i)
# train

j=0

for folder in copy_test_si:

    src='../input/rsna-str-pulmonary-embolism-detection/test/'+folder+'/'

    dst='../working/test/'+folder+'/'

    shutil.copytree(src, dst)

    j+=1

    print(j)