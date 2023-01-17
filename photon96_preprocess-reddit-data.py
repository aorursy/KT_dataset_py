# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/sarcasm/train-balanced-sarcasm.csv')
from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size=0.1)

print("Before preprocessing size train data: {}".format(len(train)))

print("Before preprocessing size test data: {}".format(len(test)))

#remove rows with comments which are no strings

train = train[train.comment.apply(lambda x: type(x) == str)]

test = test[test.comment.apply(lambda x: type(x) == str)]

print("Size train data: {}".format(len(train)))

print("Size test data: {}".format(len(test)))
#save prepared test & train data

train.to_csv('train_sarcasm', index=False)

test.to_csv('test_sarcasm', index=False)