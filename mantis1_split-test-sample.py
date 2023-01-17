import numpy as np
import pandas as pd

train = pd.read_csv('../input/train.csv', sep=',')
df_test_sample = []

df_test_sample = train[:5000]


names = train["Breed"].unique()


