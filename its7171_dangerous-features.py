import numpy as np

import matplotlib as mpl

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

import warnings

%matplotlib inline

warnings.simplefilter('ignore')



trn = pd.read_json('../input/stanford-covid-vaccine/train.json', lines=True)

tst = pd.read_json('../input/stanford-covid-vaccine/test.json', lines=True)

pri = tst[tst.seq_length == 130]

pub = tst[tst.seq_length == 107]
def read_bpps_mean(df):

    bpps_arr = []

    for mol_id in df.id.to_list():

        bpps_arr.append(np.load(f"../input/stanford-covid-vaccine/bpps/{mol_id}.npy").mean(axis=1))

    return bpps_arr



trn['bpps_mean'] = read_bpps_mean(trn)

pri['bpps_mean'] = read_bpps_mean(pri)

pub['bpps_mean'] = read_bpps_mean(pub)



sns.distplot(np.array(trn['bpps_mean'].to_list()).reshape(-1),color="Blue")

sns.distplot(np.array(pub['bpps_mean'].to_list()).reshape(-1),color="Green")

sns.distplot(np.array(pri['bpps_mean'].to_list()).reshape(-1),color="Red")
def read_bpps_max(df):

    bpps_arr = []

    for mol_id in df.id.to_list():

        bpps_arr.append(np.load(f"../input/stanford-covid-vaccine/bpps/{mol_id}.npy").max(axis=1))

    return bpps_arr



trn['bpps_max'] = read_bpps_max(trn)

pri['bpps_max'] = read_bpps_max(pri)

pub['bpps_max'] = read_bpps_max(pub)



sns.distplot(np.array(trn['bpps_max'].to_list()).reshape(-1),color="Blue")

sns.distplot(np.array(pub['bpps_max'].to_list()).reshape(-1),color="Green")

sns.distplot(np.array(pri['bpps_max'].to_list()).reshape(-1),color="Red")
def read_bpps_sum(df):

    bpps_arr = []

    for mol_id in df.id.to_list():

        bpps_arr.append(np.load(f"../input/stanford-covid-vaccine/bpps/{mol_id}.npy").sum(axis=1))

    return bpps_arr



trn['bpps_sum'] = read_bpps_sum(trn)

pri['bpps_sum'] = read_bpps_sum(pri)

pub['bpps_sum'] = read_bpps_sum(pub)



sns.distplot(np.array(trn['bpps_sum'].to_list()).reshape(-1),color="Blue")

sns.distplot(np.array(pub['bpps_sum'].to_list()).reshape(-1),color="Green")

sns.distplot(np.array(pri['bpps_sum'].to_list()).reshape(-1),color="Red")
def read_bpps_nb(df):

    bpps_arr = []

    for mol_id in df.id.to_list():

        bpps = np.load(f"../input/stanford-covid-vaccine/bpps/{mol_id}.npy")

        bpps_nb = (bpps > 0).sum(axis=0) / bpps.shape[0]

        bpps_arr.append(bpps_nb)

    return bpps_arr 



trn['bpps_nb'] = read_bpps_nb(trn)

pri['bpps_nb'] = read_bpps_nb(pri)

pub['bpps_nb'] = read_bpps_nb(pub)



sns.distplot(np.array(trn['bpps_nb'].to_list()).reshape(-1),color="Blue")

sns.distplot(np.array(pub['bpps_nb'].to_list()).reshape(-1),color="Green")

sns.distplot(np.array(pri['bpps_nb'].to_list()).reshape(-1),color="Red")
def mk_pair_map(structure, type='pm'):

    pm = np.full(len(structure), -1, dtype=int)

    pd = np.full(len(structure), -1, dtype=int)

    queue = []

    for i, s in enumerate(structure):

        if s == "(":

            queue.append(i)

        elif s == ")":

            j = queue.pop()

            pm[i] = j

            pm[j] = i

            pd[i] = i-j

            pd[j] = i-j

    if type == 'pm':

        return pm

    elif type == 'pd':

        return pd
trn['pair_map'] = trn.structure.apply(mk_pair_map, type='pm')

pub['pair_map'] = pub.structure.apply(mk_pair_map, type='pm')

pri['pair_map'] = pri.structure.apply(mk_pair_map, type='pm')



trn_list = np.array(trn['pair_map'].to_list()).reshape(-1)

pub_list = np.array(pub['pair_map'].to_list()).reshape(-1)

pri_list = np.array(pri['pair_map'].to_list()).reshape(-1)



sns.distplot(trn_list[~trn_list<0],color="Blue")

sns.distplot(pub_list[~pub_list<0],color="Green")

sns.distplot(pri_list[~pri_list<0],color="Red")
trn['pair_dist'] = trn.structure.apply(mk_pair_map, type='pd')

pub['pair_dist'] = pub.structure.apply(mk_pair_map, type='pd')

pri['pair_dist'] = pri.structure.apply(mk_pair_map, type='pd')



trn_list = np.array(trn['pair_dist'].to_list()).reshape(-1)

pub_list = np.array(pub['pair_dist'].to_list()).reshape(-1)

pri_list = np.array(pri['pair_dist'].to_list()).reshape(-1)



sns.distplot(trn_list[~trn_list<0],color="Blue")

sns.distplot(pub_list[~pub_list<0],color="Green")

sns.distplot(pri_list[~pri_list<0],color="Red")