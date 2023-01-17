import os

from nltk.parse import malt

os.listdir('/kaggle/input')
import pandas as pd

df = pd.read_csv('/kaggle/input/gendered-pronoun-resolution/test_stage_1.tsv', delimiter='\t')

df.head()
from nltk.parse import malt

import os

os.listdir('/kaggle/input')

os.environ['MALT_PARSER'] = '/kaggle/input/maltparser19'

os.environ['MALT_MODEL'] = '/kaggle/input/maltparser19/engmalt.linear-1.7.mco'

mp = malt.MaltParser( '/kaggle/input', 'engmalt.linear-1.7.mco') 

tree = mp.parse_one(df.iloc[0].Text.split())

tree
from nltk.tree import Tree

max_depth = 32

def findintree(t,d=0):

    for i in range(len(t)):

        if d<max_depth:

            if type(t[i]) is Tree:

                findintree(t[i],d+1)

            else:

                print(d,i,str(t[i]))

findintree(tree.tree())
max_path_len = 16

def search_path_for_leaf(t,kw,path,d=0):

    for i in range(len(t)):

        if d<max_depth:

            if type(t[i]) is Tree:

                for p in path.keys():

                    path[p] += 1

                search_path_for_leaf(t[i],kw,path,d+1)

            else:

                for k in kw:

                    if k in str(t[i]):

                        if k not in path.keys():

                            path[k] = 0

                        for p in list(path.keys()):

                            path[p+'-'+k] = 0

                            links = p.split('-')

                            last = links[-1]

                            bef_key = last+'-'+k

                            if bef_key in path.keys():

                                path[bef_key] = min(path[bef_key], path[p])

                            else:

                                path[bef_key] = path[p]

def get_score_index( index ):

    tree = mp.parse_one(df.iloc[index].Text.split())

    path = {}

    ap = df.iloc[index].A.split()

    bp = df.iloc[index].B.split()

    p = df.iloc[index].Pronoun.split()

    keywords = ap + bp + p

    search_path_for_leaf(tree.tree(),keywords,path)

    score_A = max_depth

    score_B = max_depth

    score_N = max_depth

    for _p in p:

        for _ap in ap:

            if _ap+'-'+_p in path.keys():

                score_A = min(score_A,path[_ap+'-'+_p])

            elif _p+'-'+_ap in path.keys():

                score_A = min(score_A,path[_p+'-'+_ap])

        for _bp in bp:

            if _bp+'-'+_p in path.keys():

                score_B = min(score_B,path[_bp+'-'+_p])

            elif _p+'-'+_bp in path.keys():

                score_B = min(score_B,path[_p+'-'+_bp])

    if max_path_len <= score_A and max_path_len <= score_B:

        score_A = 0.

        score_B = 0.

        score_N = 1.

    elif score_A == score_B:

        score_A = 0.5

        score_B = 0.5

        score_N = 0.

    elif score_A <= max_path_len:

        score_A = 0.

        score_B = 1.

        score_N = 0.

    elif score_B <= max_path_len:

        score_A = 1.

        score_B = 0.

        score_N = 0.

    else:

        _score_A = max_depth / score_A

        score_B = max_depth / score_B

        score_A = _score_A

        score_N = 0.

    return score_A, score_B, score_N

print(df.iloc[0].A,'-',df.iloc[0].B,get_score_index(0))

print(df.iloc[1].A,'-',df.iloc[1].B,get_score_index(1))

print(df.iloc[2].A,'-',df.iloc[2].B,get_score_index(2))

print(df.iloc[3].A,'-',df.iloc[3].B,get_score_index(3))

print(df.iloc[4].A,'-',df.iloc[4].B,get_score_index(4))
from multiprocessing import Pool

def multi_get_score(index):

    return index, get_score_index(idx)

with Pool(4) as p:

    result = p.map(multi_get_score, list(range(len(df))))

dst = np.zeros((len(df),3))

for index, t in result:

    dst[index][0] = t[0]

    dst[index][1] = t[1]

    dst[index][2] = t[2]

with open('submission.csv','w') as f:

    f.write('ID,A,B,NEITHER\n')

    for idx in range(len(df)):

        score = get_score_index(idx)

        f.write('%s,%f,%f,%f\n'%(df.iloc[idx].ID,dst[index][0],dst[index][1],dst[index][2]))

        if idx % 100 == 0:

            print(idx+1,'/',len(df))

print('done')