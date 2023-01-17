import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

sns.set()
date_rng = pd.date_range(start='1/1/2018', end='1/08/2018', freq='H')
df = pd.DataFrame(date_rng, columns=['date'])
df['data'] = np.random.randint(0,100,size=(len(date_rng)))
df.head(10)
df['data'].plot(figsize=(16,6))
def rec_plot(s, eps=0.10, steps=10):
    d = pdist(s[:,None])
    d = np.floor(d/eps)
    d[d>steps] = steps
    Z = squareform(d)
    return Z
a = []

a.append(rec_plot(df['data'], eps=0.10))
a.append(rec_plot(df['data'], eps=0.30))
a.append(rec_plot(df['data'], eps=0.60))
a.append(rec_plot(df['data'], eps=0.80))
a.append(rec_plot(df['data'], eps=5))


fig, axs = plt.subplots(1, len(a), figsize=(16, 4), sharey=True)

for i, obj in enumerate(a):
    axs[i].imshow(obj, cmap='gray')