# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



# Displaying all columns

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)
# Reading in the CSVs

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

submission = pd.read_csv("../input/sample_submission.csv")



SEED = 5471242
train.head()
from sklearn.model_selection import train_test_split



X = train.drop(["Malware", "Category", "Package"], axis=1)

Y = train["Malware"]



# Splitting train dataset into X_train, Y_train, X_val, Y_val



X_train, X_val, Y_train, Y_val = train_test_split(X,

                                                  Y,

                                                  test_size=0.20,

                                                  random_state=SEED)
X_train.head()
Y_train.head()