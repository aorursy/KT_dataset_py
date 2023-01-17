import numpy as np 

import pandas as pd 



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

import matplotlib.pyplot as plt
train = pd.read_csv('../input/train.csv')

train.price_doc = train.price_doc/train.full_sq
train.kremlin_km.min()
fig, ax = plt.subplots(figsize = (12,10))

ax.hist(train.loc[train.kremlin_km < 0.08, 'price_doc'], bins =200)

ax.set_title("Flats between Putin's courtyard and Lenin's Mausoleum" )

ax.set_xlim(0,500000)

ax.set_xlabel('PRICE for m2')

ax.set_ylabel('Number of observations')

plt.show()