import numpy as np

import pandas as pd

from IPython.display import display

import matplotlib.pyplot as plt



data_train = pd.read_csv("../input/train.csv")

data_test = pd.read_csv("../input/test.csv")

data_macro = pd.read_csv("../input/macro.csv")



prices_train = data_train['price_doc']

features_train = data_train.drop('price_doc', axis = 1)



prices_test = []

features_test = data_test



print (features_train.head())
plt.scatter(features_train['id'],prices_train)