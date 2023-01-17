import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv("../input/train.csv")



# Print the column labels

df.columns
df.shape
df.info()
x = np.random.normal(29.6,14.5,1000)

print(x>0)