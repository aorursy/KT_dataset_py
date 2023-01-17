import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Read data from train set and print summary

train_url = "../input/train.csv"

titanic_train = pd.read_csv(train_url)

titanic_train.describe()
test_url = "../input/test.csv"

titanic_test = pd.read_csv(test_url)

titanic_test.describe()