import random
import numpy as np
import pandas as pd
filename = "../input/train.csv"

def file_len(filename):
    with open(filename) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

n = file_len(filename)

print(n)

s = 500000
skip = sorted(random.sample(range(1, n), n-s))

train_sample = pd.read_csv(filename, skiprows=skip)
train_sample.describe()

