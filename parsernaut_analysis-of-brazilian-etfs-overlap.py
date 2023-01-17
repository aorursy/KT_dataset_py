import pandas as pd

import matplotlib.pyplot as plt

import squarify

import itertools as it
etfs = pd.read_csv('../input/b3etfs/b3Indexes.csv', sep=',')

display(etfs)

etfList = ['IBOV', 'IBRX', 'IDIV', 'IVBX', 'SMLL']
for code in etfList:

    fig, ax = plt.subplots(figsize=(16, 14))

    plt.title(code, fontweight='bold', fontsize=16)

    plt.axis('off')

    sizes = (etfs.loc[:, [code, 'Sectors', 'Market Cap (USD) 08/2020']]).groupby(['Sectors']).sum()['Market Cap (USD) 08/2020']

    label = sizes.index.values

    squarify.plot(sizes=sizes,alpha=0.7, label=label, ax=ax)

    plt.show()
weighted = etfs.set_index('Code')[etfList]

unweighted = weighted.where(weighted < 1e-4, 1)

display(weighted, unweighted)
pairList = it.combinations(etfList, 2)

overlap = pd.DataFrame(index=pairList)

overlap.loc[:, 'Number of shared stocks'] = [unweighted[item[0]].dot(unweighted[item[1]]) for item in overlap.index]

overlap.loc[:, '% of shared in A'] = [round(unweighted[item[0]].dot(unweighted[item[1]]) / unweighted[item[0]].sum() * 100, 0) for item in overlap.index]

overlap.loc[:, '% of shared in B'] = [round(unweighted[item[0]].dot(unweighted[item[1]]) / unweighted[item[1]].sum() * 100, 0) for item in overlap.index]

overlap.loc[:, '% Overlap by Weight'] = [round(0.5 * (weighted[item[0]].dot(unweighted[item[1]]) + weighted[item[1]].dot(unweighted[item[0]])), 2) for item in overlap.index]

overlap
pairList = it.combinations(etfList, 2)

sectorVal = etfs.set_index('Sectors')[etfList].groupby('Sectors').sum()

overlap = pd.DataFrame(columns=list(pairList), index=sectorVal.index)

for item in overlap.columns:

    overlap[item] = sectorVal[item[0]].values - sectorVal[item[1]].values

overlap
unweighted[unweighted.sum(axis=1)==len(etfList)]
display(unweighted[unweighted.sum(axis=1)==1])

unweighted[unweighted.sum(axis=1)==1].sum()