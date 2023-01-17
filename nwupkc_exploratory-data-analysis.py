# import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import data
df = pd.read_csv('../input/honeyproduction.csv')
df.info()
df.head()
df.groupby('year').yieldpercol.mean().plot(figsize=(10,6), title='Average Yield Change From 1998 To 2012');
f, ax = plt.subplots(len(df.state.unique())//10+1, sharex=True, sharey=True, figsize=(12,20))
for i in range(0, len(df.state.unique())//10+1):
    for s in df.state.unique()[10*i:10*(i+1)]:
        ax[i].plot('year', 'yieldpercol', data=df[df.state==s])
    ax[i].legend(df.state.unique()[10*i:10*(i+1)], bbox_to_anchor=(1.04,1), loc="upper left")
# Top honey producers over time
df.loc[df.groupby(["year"])["yieldpercol"].idxmax(), ['year', 'state', 'yieldpercol']].set_index('year')
# Least producers over time
df.loc[df.groupby(["year"])["yieldpercol"].idxmin(), ['year', 'state', 'yieldpercol']].set_index('year')
df['diff'] = df.groupby('state')['yieldpercol'].diff()
df['absdiff'] = df['diff'].abs()
df.loc[df.groupby(["year"])["absdiff"].idxmax(), ['year', 'state', 'diff']].set_index('year')
df['percdiff'] = df['diff']/df['yieldpercol']*100
df['abspercdiff'] = df['percdiff'].abs()
df.loc[df.groupby(["year"])["abspercdiff"].idxmax(), ['year', 'state', 'percdiff']].set_index('year')
df[df.year<2006].groupby('year').numcol.sum().plot();
df.groupby('year').numcol.sum().plot();
df[df.year<2006].groupby('year').yieldpercol.mean().plot();
df.groupby('year').yieldpercol.mean().plot();
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
ax1.plot(df.groupby('year').totalprod.sum())
ax2.plot(df.groupby('year').prodvalue.mean());