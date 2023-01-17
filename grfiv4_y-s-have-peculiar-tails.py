import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
with pd.HDFStore("../input/train.h5", "r") as train:

    d = train.get("train")
plt.figure(figsize=(12,12))

sns.distplot(d.y)

plt.axvline(x=d.y.mean()-2*d.y.std())

plt.axvline(x=d.y.mean()+2*d.y.std())

plt.show()