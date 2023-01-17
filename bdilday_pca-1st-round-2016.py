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

import numpy as nppr
import re
import copy
import glob
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import seaborn as sns

fdir = '../input'

teams_to_seed = {1276: 'W11', 1195: 'W16', 1455: 'Y11', 1221: 'Z16'}

def chomp(f):
    lines = [l.strip() for l in open('{}/{}'.format(fdir, f)).readlines()]
    return [l for l in lines[1:] if len(l)>0]

def get_probabilities():
    data = []
    fs = glob.glob('{}/predictions/*csv'.format(fdir))

    for f in fs:
        name = f.split('/')[-1].split('.csv')[0]
        tmp = []
        lines = chomp(f)
        for l in lines:
            gid, p = l.split(',')
            try:
                p = float(p)
            except ValueError:
                gid, p, = p, gid
                p = float(p)

            tmp.append(tuple([name, gid, p]))
        tmp.sort()
        data.append(tmp)
    return np.array(data, dtype=[('name', 'U64'), ('id', 'U16'),('pred', 'f8')])

def make_data():
    teams = {}
    f = 'Teams.csv'
    lines = chomp(f)
    for l in lines:
        id, name = l.split(',')
        id = int(id)
        teams[id] = name
        teams[name] = id

    seeds = {}
    f = 'TourneySeeds.csv'
    lines = chomp(f)
    for l in lines:
        yr, sd, t = l.split(',')
        yr = int(yr)
        t = int(t)
        if not yr in seeds:
            seeds[yr] = {}
        if yr==2016 and t in teams_to_seed:
            sd = teams_to_seed[t]
        seeds[yr][t] = sd
        seeds[yr][sd] = t
    # play-in for 2016

    data = {}
    slots = {}
    f = 'TourneySlots.csv'
    lines = chomp(f)
    for l in lines:
        yr, slot, strongseed, weakseed = l.split(',')
        yr = int(yr)

        if yr != 2016:
            continue

        if not yr in slots:
            slots[yr] = {}
        if not yr in data:
            data[yr] = []

        m = re.match('R([1-9]{1})([A-Z]{1})([1-9]{1})', slot)
        if m:
            rnd, region, game = m.groups()
            tmp = {'round': rnd, 'region': region, 'game': game}

            if not re.match('^[A-Z]{1}[0-9]{2}', strongseed):
                continue

            t1 = seeds[yr][strongseed]
            t2 = seeds[yr][weakseed]
            if t2<t1:
                t1, t2 = t2, t1
            tmp['t1'] = t1
            tmp['t2'] = t2
            tmp['name1'] = teams[t1]
            tmp['name2'] = teams[t2]

            data[yr].append(tmp)

    data['matchups'] = {}
    for d in data[2016]:
        k = '2016_{}_{}'.format(d['t1'], d['t2'])
        data['matchups'][k] = copy.copy(d)
    return teams, seeds, data

teams, seeds, data = make_data()
all_probs = get_probabilities()
ks = data['matchups'].keys()
ks = sorted([k for k in ks])
X = []
for k in ks:
    cc = np.where(all_probs['id'] == k)
    X.append(all_probs[cc]['pred'])
X = np.array(X)
pca = PCA(n_components=32)
pca.fit(X.transpose())
idxs = np.argsort((pca.components_[0:10,:]**2).sum(0))[::-1]
for i in range(32):
    plt.plot(pca.components_[:,idxs[i]] - i , drawstyle='steps-mid', color='k')
for i, _ in enumerate(ks):
    k = ks[idxs[i]]
    plt.text(-0.1, -i, '{} - {}'.format(data['matchups'][k]['name1'],
                                        data['matchups'][k]['name2']),
                                        horizontalalignment='right', fontsize=6)
plt.xticks([])
plt.yticks([])
plt.xlim(-8, 32)
plt.ylim(-33, 2)
plt.title('PCA of 1st round predictions - 2016')
plt.xlabel('Principal component number')
plt.savefig('PCA_1st_2016.png')
#plt.show()

