import pandas as pd

import matplotlib.pyplot as plt



import os

print(os.listdir("../input"))



%matplotlib inline
test_df = pd.read_csv("../input/train.csv",nrows=5)

test_df.head()
test_df.dtypes
df = pd.read_csv("../input/train.csv")

df.head()
df.describe().round(2)
df['Age'].plot.hist()