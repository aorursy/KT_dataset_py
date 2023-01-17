# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt 

from matplotlib.ticker import FuncFormatter as ff

import matplotlib as mpl

mpl.rcParams['lines.linewidth'] = 4

%matplotlib inline

g = pd.read_csv(r'../input/global.csv')

r = pd.read_csv(r'../input/regional.csv')

g.set_index('year', inplace=True)

g = g[g.index>1945]

r.set_index('year', inplace=True) 

r = r[r.index>1945]
relis_num = [a for a in g.columns if (a.find('_all')!=-1) & (a.find('religion_all')!=0)]

relis_pct =[

 'christianity_percent',

 'judaism_percent',

 'islam_percent',

 'buddhism_percent',

 'zoroastrianism_percent',

 'hinduism_percent',

 'sikhism_percent',

 'shinto_percent',

 'baha’i_percent',

 'taoism_percent',

 'jainism_percent',

 'confucianism_percent',

 'syncretism_percent',

 'animism_percent',

 'otherreligion_percent', 'noreligion_percent'] 

import seaborn as sns

fig, ax = plt.subplots(figsize=(10,6.2))



current = g.index.max()

ax.set_title('largest current religions')

recent = (g[g.index==current][relis_num].T).sort_values(by=current).iloc[-6:]

recent.T.plot(kind='bar', ax=ax, legend=False)

ax.legend([i.split('_')[0] for i in recent.index])

ax.get_yaxis().set_major_formatter(ff(lambda x, p: format(int(x/1000000), ',')))

ax.set_ylabel('believers in millions')

plt.tight_layout()

plt.show()

main_relis_num = list(recent.index)

main_relis_pct = [a.split('_')[0]+'_'+'percent' for a in recent.index]
fig, ax = plt.subplots(figsize=(10,6.2))

g[main_relis_pct].sum(axis=1).plot(ax=ax)

ax.get_yaxis().set_major_formatter(ff(lambda x, p: format('{:3.0%}'.format(x))))
fig, ax = plt.subplots(figsize=(10,6.2))

for i in main_relis_num: ax.plot(g[i], label=i.split('_')[0])

ax.set_title('Overview')

ax.legend([i.split('_')[0] for i in main_relis_num])

ax.get_yaxis().set_major_formatter(ff(lambda x, p: format(int(x/1000000), ',')))

ax.set_ylabel('believers in millions')

plt.tight_layout()





plt.show()
fig, ax = plt.subplots(figsize=(10,6.2))

for i in main_relis_pct: ax.plot(g[i], label=i.split('_')[0])

ax.set_title('Overview (pct of World population)')

ax.legend([i.split('_')[0] for i in main_relis_pct])

ax.get_yaxis().set_major_formatter(ff(lambda x, p: '{:3.0%}'.format(x)))

plt.tight_layout()

plt.show()

colors = {a.split('_')[0]:'k' for a in relis_num}

for a in ax.get_lines(): colors [a.get_label()] = a.get_color() 
gr = r.groupby('region')

fig, aa = plt.subplots(3,2, figsize=(10,6.2))

i=1

for re, dre in gr:

    ax = aa[np.mod(i,3), i//3]

    i += 1

    d = dre[main_relis_pct]

    d = d.apply(lambda x: x.loc[2010]-x.loc[1985])

    d.sort_values(inplace=True)

    d.plot(kind='barh', ax=ax, color=[colors[a.split('_')[0]] for a in d.index])

    #ax.set_yticklabels([str(a.get_text()).split('_')[0] for a in ax.get_yticklabels()])

    ax.set_yticklabels([])

    ax.set_title(re)

    ax.get_xaxis().set_major_formatter(ff(lambda x, p: '{:3.1%}'.format(x)))

i=0

ax = aa[np.mod(i,3), i//3]

d = g[main_relis_pct].apply(lambda x: x.loc[2010]-x.loc[1985])

d.sort_values(inplace=True)

d.plot(kind='barh', ax=ax, color=[colors[a.split('_')[0]] for a in d.index])

ax.set_yticklabels([str(a.get_text()).split('_')[0] for a in ax.get_yticklabels()])

ax.set_title('Global')

ax.get_xaxis().set_major_formatter(ff(lambda x, p: '{:3.1%}'.format(x)))

plt.tight_layout()

plt.show()
fig, ax = plt.subplots(figsize=(10, 6.2))

d = gr.get_group('Africa')[relis_pct].apply(lambda x: x.loc[2010]-x.loc[1985]).sort_values()

d = d[d.abs()>0.005]

d.plot(kind='barh', ax=ax, color=[colors[a.split('_')[0]] for a in d.index])

ax.set_yticklabels([str(a.get_text()).split('_')[0] for a in ax.get_yticklabels()])

ax.get_xaxis().set_major_formatter(ff(lambda x, p: '{:3.0%}'.format(x)))

plt.tight_layout()

plt.show()
popus = r.reset_index().groupby(['year', 'region']).population.sum().unstack()

fig, aa = plt.subplots(2,1, figsize=(10,12.4))

ax=aa[0]

ax.plot(popus.iloc[-6:]/1000000)

ax.plot(g.iloc[-6:].population/1000000)

ax.set_ylabel('Population in millions')

ax.legend(list(popus.columns) + ['Global'])

ax=aa[1]

ax.semilogy(popus.iloc[-6:]/popus.iloc[-6])

ax.semilogy(g.iloc[-6:].population/g.population.iloc[-6])

ax.set_ylabel('Population Growth')

plt.tight_layout()

plt.show()
popus = r.reset_index().groupby(['year', 'region']).population.sum().unstack()

popus = popus[(popus.index>=1985)]

popus_mean = popus.mean(axis=0)/1000000

popus_mean['Global'] = g['population'].mean()/1000000

popus_growth = (popus.loc[2010]/popus.loc[1985])**(1/(len(popus)*5))-1

popus_growth['Global'] = (g.loc[2010,'population']/g.loc[1985,'population'])**(1/(len(g)*5))-1

fig, aa = plt.subplots(2,1, figsize=(10,12.4))

ax=aa[0]

popus_mean.sort_values().plot(kind='barh', ax=ax)

ax.set_xlabel('Mean Population in millions')

ax.legend(popus.columns)

ax=aa[1]

popus_growth.sort_values().plot(kind='barh', ax=ax)

ax.set_xlabel('Population Annual Growth')

ax.get_xaxis().set_major_formatter(ff(lambda x, p: '{:3.1%}'.format(x)))

plt.tight_layout()

plt.show()
gr = r.groupby('region')

fig, aa = plt.subplots(3,2, figsize=(10,12.4))

i=1

for re, dre in gr:

    ax = aa[np.mod(i,3), i//3]

    i += 1

    d = dre[main_relis_pct].multiply(dre.worldpopulation_percent, axis=0)

    d = d.apply(lambda x: x.loc[2010]-x.loc[1985])

    d.sort_values(inplace=True)

    d.plot(kind='barh', ax=ax, color=[colors[a.split('_')[0]] for a in d.index])

    #ax.set_yticklabels([str(a.get_text()).split('_')[0] for a in ax.get_yticklabels()])

    ax.set_yticklabels([])

    ax.set_title(re)

    ax.get_xaxis().set_major_formatter(ff(lambda x, p: '{:3.1%}'.format(x)))

i=0

ax = aa[np.mod(i,3), i//3]

d = g[main_relis_pct].apply(lambda x: x.loc[2010]-x.loc[1985])

d.sort_values(inplace=True)

d.plot(kind='barh', ax=ax, color=[colors[a.split('_')[0]] for a in d.index])

ax.set_yticklabels([str(a.get_text()).split('_')[0] for a in ax.get_yticklabels()])

ax.set_title('Global')

ax.get_xaxis().set_major_formatter(ff(lambda x, p: '{:3.1%}'.format(x)))

plt.tight_layout()

plt.show()