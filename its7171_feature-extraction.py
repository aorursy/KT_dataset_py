import pandas as pd

import numpy as np

from collections import deque, defaultdict
# additional features



# sum of bpps

def read_bpps_sum(df):

    bpps_arr = []

    for mol_id in df.id.to_list():

        bpps_arr.append(np.load(f"../input/stanford-covid-vaccine/bpps{bpps_engine}/{mol_id}.npy").max(axis=1))

    return bpps_arr



# sum value of bpps

def read_bpps_max(df):

    bpps_arr = []

    for mol_id in df.id.to_list():

        bpps_arr.append(np.load(f"../input/stanford-covid-vaccine/bpps{bpps_engine}/{mol_id}.npy").sum(axis=1))

    return bpps_arr



# non zero number of bpps

def read_bpps_nb(df, thre=0):

    bpps_arr = []

    for mol_id in df.id.to_list():

        bpps = np.load(f"../input/stanford-covid-vaccine/bpps{bpps_engine}/{mol_id}.npy")

        bpps_arr.append((bpps > thre).sum(axis=0) / bpps.shape[0])

    return bpps_arr 



# normalization 

def norm_arr(train_arr, test_arr):

    arr1 = np.array([])

    for arr in train_arr.to_list() + test_arr.to_list():

        arr1 = np.append(arr1,arr)

    arr1_mean = arr1.mean()

    arr1_std = arr1.std()

    train_arr = (train_arr - arr1_mean) / arr1_std

    test_arr = (test_arr - arr1_mean) / arr1_std

    return train_arr, test_arr



# calclate distance of the paired nucleotide

def mk_pair_distance(structure):

    pd = np.full(len(structure), -1, dtype=int)

    start_token_indices = []

    for i, token in enumerate(structure):

        if token == "(":

            start_token_indices.append(i)

        elif token == ")":

            j = start_token_indices.pop()

            pd[i] = i-j

            pd[j] = i-j

    return pd



# get position of the paired nucleotide

def mk_pair_map(structure):

    pm = np.full(len(structure), -1, dtype=int)

    start_token_indices = []

    for i, token in enumerate(structure):

        if token == "(":

            start_token_indices.append(i)

        elif token == ")":

            j = start_token_indices.pop()

            pm[i] = j

            pm[j] = i

    return pm



# get probability of the paired nucleotide

def mk_pair_prob(arr):

    structure = arr.structure

    mol_id = arr.id

    pm = np.full(len(structure), -1, dtype=int)

    start_token_indices = []

    for i, token in enumerate(structure):

        if token == "(":

            start_token_indices.append(i)

        elif token == ")":

            j = start_token_indices.pop()

            pm[i] = j

            pm[j] = i

    bpps = np.load(f"../input/stanford-covid-vaccine/bpps{bpps_engine}/{mol_id}.npy")

    pp = np.full(len(structure), 0, dtype=float)

    for i in range(len(structure)):

        j = pm[i]

        if j >= 0:

            pp[i] = bpps[i,j]

    return pp



# get sequence of the paired nucleotide

def mk_pair_acgu(arr):

    pacgu = ['.']*len(arr.sequence)

    start_token_indices = []

    for i, (seq, token) in enumerate(zip(arr.sequence, arr.structure)):

        if token == "(":

            start_token_indices.append(i)

        elif token == ")":

            j = start_token_indices.pop()

            pacgu[i] = arr.sequence[j]

            pacgu[j] = arr.sequence[i]

    return "".join(pacgu)



# get base of the paired nucleotide

acgu2_dict = {}

def mk_pair_acgu2(arr):

    pacgu = ['..']*len(arr.sequence)

    start_token_indices = []

    for i, (seq, token) in enumerate(zip(arr.sequence, arr.structure)):

        if token == "(":

            start_token_indices.append(i)

        elif token == ")":

            j = start_token_indices.pop()

            pacgu[i] = arr.sequence[i]+arr.sequence[j]

            pacgu[j] = arr.sequence[j]+arr.sequence[i]

            acgu2_dict[pacgu[i]] = 1

            acgu2_dict[pacgu[j]] = 1

    return pacgu



# helper func: get idx value of the list

def get_list_item(sequence, idx, default='-'):

    if idx < 0 or idx >= len(sequence):

        return default

    else:

        return sequence[idx]



# getting information on the neighbors of the pair

def add_pair_feats(df):

    nseq = 5

    feats_list = []

    for idx, row in df.iterrows():

        length = len(row["structure"])

        pair_idxs = {}

        idx_stack = deque()

        for i, struct in enumerate(row["structure"]):

            if struct == "(":

                idx_stack.append(i)

            elif struct == ")":

                start = idx_stack.pop()

                pair_idxs[start] = i

                pair_idxs[i] = start



        feats = defaultdict(list)

        for i in range(length):

            pair_idx = pair_idxs.get(i)

            if pair_idx is not None:

                for k in range(-nseq, nseq+1):

                    if k == 0:

                        continue

                    feats[f"pair_seq_{k}"].append(

                        get_list_item(row["sequence"], pair_idx + k)   # basically index access with default value if out of bounds

                    )



                    feats[f"pair_strct_{k}"].append(

                        get_list_item(row["structure"], pair_idx + k)

                    )



                    feats[f"pair_loop_{k}"].append(

                        get_list_item(row["predicted_loop_type"], pair_idx + k)

                    )



            else:

                for k in range(-nseq, nseq+1):

                    if k == 0:

                        continue

                    feats[f"pair_seq_{k}"].append("-")

                    feats[f"pair_strct_{k}"].append("-")

                    feats[f"pair_loop_{k}"].append("-")

            #feats_list.append(feats)

        for k in feats:

            feats[k] = "".join(feats[k])

        feats_list.append(dict(feats))

    return pd.DataFrame(feats_list)



def mk_feats(train,test):

    # The value of bpps of the pair - the strength of the pair.

    train['pair_pp'] = train[['id','structure']].apply(mk_pair_prob, axis=1)

    test['pair_pp'] = test[['id','structure']].apply(mk_pair_prob, axis=1)

    train['pair_pp'], test['pair_pp'] = norm_arr(train['pair_pp'], test['pair_pp'])

    

    # paired sequence

    train['pair_acgu'] = train[['sequence','structure']].apply(mk_pair_acgu, axis=1)

    test['pair_acgu'] = test[['sequence','structure']].apply(mk_pair_acgu, axis=1)

    

    # the set of the pair (CG or GU or AU or None)

    train['pair_acgu2'] = train[['sequence','structure']].apply(mk_pair_acgu2, axis=1)

    test['pair_acgu2'] = test[['sequence','structure']].apply(mk_pair_acgu2, axis=1)

    

    # sum

    train['bpps_sum'] = read_bpps_sum(train)

    test['bpps_sum'] = read_bpps_sum(test)

    train['bpps_sum'], test['bpps_sum'] = norm_arr(train['bpps_sum'], test['bpps_sum'])

    

    # max

    train['bpps_max'] = read_bpps_max(train)

    test['bpps_max'] = read_bpps_max(test)

    train['bpps_max'], test['bpps_max'] = norm_arr(train['bpps_max'], test['bpps_max'])

    

    # non zero number

    train['bpps_nb'] = read_bpps_nb(train)

    test['bpps_nb'] = read_bpps_nb(test)

    train['bpps_nb'], test['bpps_nb'] = norm_arr(train['bpps_nb'], test['bpps_nb'])

    

    # more than 0.05 number

    train['bpps_nb005'] = read_bpps_nb(train, 0.05)

    test['bpps_nb005'] = read_bpps_nb(test, 0.05)

    train['bpps_nb005'], test['bpps_nb005'] = norm_arr(train['bpps_nb005'], test['bpps_nb005'])

    

    # bpps_sum-max

    train['bpps_sum-max'] = train['bpps_sum'] - train['bpps_max']

    test['bpps_sum-max'] = test['bpps_sum'] - test['bpps_max']

    train['bpps_sum-max'], test['bpps_sum-max'] = norm_arr(train['bpps_sum-max'], test['bpps_sum-max'])

    

    # Information on the neighbors of the pair

    train = pd.concat([train, add_pair_feats(train)], axis=1)

    test = pd.concat([test, add_pair_feats(test)], axis=1)

    

    return train, test
bpps_engine = ''

train = pd.read_json('../input/stanford-covid-vaccine/train.json', lines=True)

test = pd.read_json('../input/stanford-covid-vaccine/test.json', lines=True)



train, test = mk_feats(train,test)
train.head(1).T