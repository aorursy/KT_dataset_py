import numpy as np 

import pandas as pd



import os

print(os.listdir("../input"))
train = pd.read_csv("../input/train.csv")

train.head()
test = pd.read_csv("../input/test.csv")

test.head()