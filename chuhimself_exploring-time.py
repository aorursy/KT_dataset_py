import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline


train = pd.read_csv('../input/train.csv')
train.groupby("place_id").value_counts()
#get 1% of samples
df = train.sample( frac = 0.01)

df.info()

df.groupby("place_id")
