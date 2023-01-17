import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np
df = pd.read_csv('../input/pres16results.csv')

df.head()
df = df[(df['cand'] == 'Donald Trump') | (df['cand'] == 'Hillary Clinton')].dropna(subset=['county'], axis='rows')
df['county_st'] = df['county'] + ' in ' + df['st']
df.head()
piv = df.pivot(index='county_st', columns='cand', values='votes')
piv['diff'] = piv['Hillary Clinton'] - piv['Donald Trump']
piv.head()
def HCDT(diff):

    if diff > 0:

        res = 'Hillary Clinton'

    else:

        res = 'Donald Trump'

    return res
piv['Winner'] = list(map(HCDT, piv['diff']))
piv['Absolute Vote Difference'] = abs(piv['diff'])
piv.head()
g_violin = sns.catplot(x="Winner", y="Absolute Vote Difference", kind="violin", inner=None, data=piv, cut=0)

g_violin.fig.set_size_inches(6,24)

#plt.savefig('graphs/gerrymandering.jpg', dpi=600) #graph output
#fig, ax = plt.subplots(figsize=(6,24))

g_box = sns.catplot(x="Winner", y="Absolute Vote Difference", kind="box", dodge=False, data=piv, width=0.6, showfliers=False)

g_box.fig.set_size_inches(6,24)

#plt.savefig('graphs/gerrymandering_box.jpg', dpi=600) #graph output
def percentile(n):

    def percentile_(x):

        return np.percentile(x, n)

    percentile_.__name__ = 'percentile_%s' % n

    return percentile_



comp = piv[['Absolute Vote Difference', 'Winner']].groupby('Winner').agg(['min', 'max', 'mean', percentile(25), 'median', percentile(75)])

comp
piv_p = df.pivot(index='county_st', columns='cand', values='pct')

piv_p['% diff'] = piv_p['Hillary Clinton'] - piv_p['Donald Trump']

piv_p['Winner'] = list(map(HCDT, piv_p['% diff']))

piv_p['Absolute % Difference'] = abs(piv_p['% diff'])

piv_p.head()
g_violin = sns.catplot(x="Winner", y="Absolute % Difference", kind="violin", data=piv_p, cut=0)

g_violin.fig.set_size_inches(6,12)
g_box = sns.catplot(x="Winner", y="Absolute % Difference", kind="box", dodge=False, data=piv_p, width=0.6, showfliers=False)

g_box.fig.set_size_inches(6,12)