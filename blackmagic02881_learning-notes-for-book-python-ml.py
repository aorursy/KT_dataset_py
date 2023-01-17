# original code from https://github.com/rasbt/python-machine-learning-book/blob/master/code/ch02/ch02.ipynb

import pandas as pd

df = pd.read_csv("../input/Iris.csv")
df.head(n=3)
df.count()
import matplotlib.pyplot as plt
import numpy as np

# select setosa and versicolor
y = df.iloc[0:100, 5].values
print(y)
