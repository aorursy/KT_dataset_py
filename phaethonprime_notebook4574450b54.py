%pylab inline

import pandas as pd

import numpy as np

import seaborn as sns
df = pd.read_csv("../input/most_backed.csv")

df['percent'] = df['amt.pledged']/df.goal
df.columns
df.currency.value_counts()
fig, ax = pylab.subplots();

df[df.currency=='usd']['goal'].hist(bins=10**np.linspace(0, 10, 30), ax=ax);

ax.set_xscale('log'); ax.set_xlabel("Goal (USD)");