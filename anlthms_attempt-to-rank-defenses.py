import numpy as np

import pandas as pd





acc = pd.read_csv('../input/accuracy_matrix.csv', index_col=0)

err = pd.read_csv('../input/error_matrix.csv', index_col=0)

acc.values[np.where((acc.values + err.values) != 1000)] = 0

defs = acc.index

def_scores = acc.values.sum(axis=1)



topk = np.argsort(def_scores)[-10:][::-1]

def_res = pd.read_csv('../input/defense_results.csv')

def_name_map = {def_res['KaggleTeamId'][i]: def_res['TeamName'][i] for i in range(def_res.shape[0])}



print('TeamID \tTeamName  \tScore')

for ident, score in zip(defs[topk], def_scores[topk]):

    print('%s \t%s  \t%d' % (ident, def_name_map[ident], score))