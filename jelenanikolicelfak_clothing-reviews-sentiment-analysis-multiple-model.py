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
# Učitavanje svih podataka i prikaz prvih 5 redova
data = pd.read_csv("../input/womens-ecommerce-clothing-reviews/Womens Clothing E-Commerce Reviews.csv")
data.head()
# broj redova u csv fajlu
print(len(data))
# broj kolona u csv fajlu
print(len(data.columns))
data_for_train = pd.read_csv("../input/womens-ecommerce-clothing-reviews/Womens Clothing E-Commerce Reviews.csv")

# Kreiranje fajla za treniranje modela

data_for_train.drop(data_for_train.tail(5486).index,inplace=True) # drop poslednjih 5486 redova
keep_col = ['Clothing ID','Review Text','Rating']
train_file = data_for_train[keep_col]
train_file.to_csv("train_file.csv", index=False)
print(len(train_file))
train_file.head()

# Kreiranje fajla za testiranje modela
data.drop(data.head(18000).index,inplace=True) # drop prvih 18000 redova
keep_col = ['Clothing ID','Review Text']
test_file = data[keep_col]
test_file.to_csv("test_file.csv", index=False)
print(len(test_file))
test_file.head()


