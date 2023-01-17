import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



train_df = pd.read_csv("../input/train.csv")

test_df  = pd.read_csv("../input/test.csv")



train_df.sample(3)

train_df.info()