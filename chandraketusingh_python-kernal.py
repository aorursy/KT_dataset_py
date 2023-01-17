import pandas as pd

import numpy as np

import os

print(os.listdir("../input"))
Rest = pd.read_csv('../input/zomato.csv', encoding='latin-1')

Rest.head()

Rest.shape
Rest.to_csv("Rest.csv")
