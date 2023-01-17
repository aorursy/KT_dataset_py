import numpy as np 
import os

print(os.listdir("../input"))
import pickle

def unpickle_me(filename):
    with open(filename, 'rb') as fh:
        return pickle.load(fh)
    
instruments = unpickle_me('../input/instruments.p')
print(type(instruments))
print(instruments.shape)
print(instruments.columns.values)

    
instruments.head(100)
import matplotlib.pyplot as plt
import seaborn as sns
fig, ax = plt.subplots()
sns.regplot(x='num_employees', y='market_cap', data=instruments, dropna=True)
plt.show()

fig, ax = plt.subplots()
ax.set(xscale="log", yscale="log")
sns.regplot(x='num_employees', y='market_cap', data=instruments, dropna=True)
plt.show()

fig, ax = plt.subplots()
ax.set(xscale="log", yscale="log")
sns.regplot(x='shares_outstanding', y='market_cap', data=instruments, dropna=True)
plt.show()

fig, ax = plt.subplots()
ax.set(xscale="linear", yscale="linear")
year_no_na = instruments['year_founded'].dropna()
bins = int(max(year_no_na) - min(year_no_na))
sns.distplot(a=year_no_na, bins=bins, kde=False)

from collections import Counter

year_counts = Counter(year_no_na)
for k,v in sorted(year_counts.items(), key=lambda x: x[1], reverse=True):
    print("{}: {}".format(int(k), v))
instruments.loc[instruments['year_founded']==2013].head(53)

instruments.loc[instruments['year_founded']==1997].head(50)

fig, ax = plt.subplots()
ax.set(xscale="linear", yscale="log")
sns.regplot(x='year_founded', y='market_cap', data=instruments, dropna=True)
plt.show()

with_history = instruments.loc[~instruments.payout_history.isnull()]
print(with_history.shape)
with_history.head()
ex_history = with_history.iloc[40].loc['payout_history']
print(type(ex_history))
print(ex_history.shape)
ex_history.head()
