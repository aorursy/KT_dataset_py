!conda install -y -c bioconda forgi;

!conda install -y -c bioconda viennarna;



import forgi.graph.bulge_graph as fgb 

import forgi.visual.mplotlib as fvm



import numpy as np

import pandas as pd



train = pd.read_json('../input/stanford-covid-vaccine/train.json', lines=True)

test = pd.read_json('../input/stanford-covid-vaccine/test.json', lines=True)
print(' Train shape: ', train.shape, '\n', 'Test shape: ', test.shape)
train.info()

train.head(1)
test.info()

test.head(1)
print('\n',train.iloc[0].sequence, '\n\n', train.iloc[0].structure, '\n\n', train.iloc[0].predicted_loop_type,)
train['seqpair']=[train.iloc[i]['structure'].replace('.', 'X') for i in range(len(train))]

train['seqpairpos']=None

train['seqpairpos_dist']=None



def Sort_Tuple(tup):  

    ##i/p: list of tuples. o/p: sorted list

    tup.sort(key = lambda x: x[0])  

    return tup  

    ##strange why return tup.sort(key = lambda x: x[0]) does not work



def pairup(row):

    ##take each row and pair the bases, create positions maps also

    queue = []

    seqpair=list(row.seqpair)

    seqpairposmap = []

    seqpairpos_dist = []



    for i in range(len(row.structure)) :

        if row.structure[i]=='(':

            queue.append(i) 

        if row.structure[i]==')':

            pairpos = queue.pop()

            ##i'th base is paired with pairpos'th base

            seqpair[pairpos] = row.sequence[i]

            seqpair[i] = row.sequence[pairpos]

            seqpairposmap.append((i,pairpos))

            seqpairposmap.append((pairpos,i))

        if row.structure[i]=='.':

            ##unpaired bases. They map to themselves

            seqpairposmap.append((i,i))

 

    train.loc[train.id==row.id, 'seqpairpos'] = [Sort_Tuple(seqpairposmap)]

    ##If you have strong masachotic tendencies :) try - Sort_Tuple(seqpairpos) without [] braces

    ##for some reason the loop repeats twice for every record

    

    ##Also find distance between pairs

    explodedpair=train[train.id==row.id]['seqpairpos'].explode().tolist()

    [seqpairpos_dist.append(abs(explodedpair[counter][1]-explodedpair[counter][0])) for counter in range (len(explodedpair))]



    train.loc[train.id==row.id, 'seqpair'] = ''.join(seqpair)

    ##An ugly hack. Let me know if there is a better way. Pandas just does not want to store lists in cells

    train.loc[train.id==row.id, 'seqpairpos_dist'] = pd.Series([seqpairpos_dist] * len(train))



train.apply(pairup, axis=1)

print('\n',train.iloc[0].sequence, '\n', train.iloc[0].structure, '\n', train.iloc[0].predicted_loop_type, '\n', train.iloc[0].seqpair, '\n\nDistance\n', train.iloc[0].seqpairpos_dist, '\n\nPairPos\n', train.iloc[0].seqpairpos)
import matplotlib.pyplot as plt

from pathlib import Path

   

bppm=np.load(Path("../input/stanford-covid-vaccine/bpps") / f"{train.loc[0].id}.npy")

plt.figure(figsize = (12, 8))

plt.imshow(bppm)

plt.colorbar()
##Set all zeros

predicted = np.zeros((len(train.iloc[0].structure), len(train.iloc[0].structure)))

pairs = []



##Set BPPM=1 for all paired bases as given in Structure column

for i, structure in enumerate(train.iloc[0].structure):

    if structure == "(":

        pairs.append(i)

    elif structure == ")":

        j = pairs.pop()

        predicted[i, j] = 1

        predicted[j, i] = 1



f, axarr = plt.subplots(1,2,figsize = (20, 10))

axarr[0].imshow(bppm)

axarr[1].imshow(predicted)
plt.figure(figsize = (18, 18))

fvm.plot_rna(fgb.BulgeGraph.from_fasta_text(f'>rna1\n{train.iloc[0].structure}\n{train.iloc[0].sequence}')[0])

plt.show()



pd.set_option('display.max_colwidth', -1)

print('\n',train.iloc[0].sequence, '\n', train.iloc[0].structure, '\n', train.iloc[0].predicted_loop_type, '\n\n', \

      train.iloc[0].seqpair, '\n\nDistance\n', train.iloc[0].seqpairpos_dist, '\n\nPairPos\n', train.iloc[0].seqpairpos)

pd.reset_option('display.max_colwidth')
rc_dict=[]

train['rc']=train['sequence']



def findRC(row):

    queue = []

    restartfrom=0

    token=''

    tok=0

    rc=row.rc

                                    

    for i in range(len(row.structure)) :

        if restartfrom>i:

            continue

    

        if row.structure[i]=='(':

            queue.append(i) 

    

        if row.structure[i]==')':

            for j in range(i,len(row.structure)):

                if row.structure[j] != ')':

                    break;



            length=j-i

            for a in range(j-i):

                b = queue.pop()

            if length>1:

                ##print('seg start:', b+1, ' seg length:', length, 'seg:',row.sequence[b:b+length])

                ##print('rseg start:', i+1, ' rseg length:', length, 'rseg:',row.sequence[i:i+length], '\n')

                rc=rc[0:b]+''.join([token+str(tok) for x in range(length)])+rc[b+length:i]+''.join([token+str(tok) for x in range(length)])+rc[i+length:]

                tok=tok+1

                restartfrom=i+length

                rc_dict.append(row.sequence[b:b+length])



    ##print(rc)

    print('More than 10 tokens') if (tok-1)>9 else None

    train.loc[train.id==row.id, 'rc'] = rc

    

train.apply(findRC, axis=1)

print('\n',train.iloc[0].sequence, '\n', train.iloc[0].structure, '\n', train.iloc[0].predicted_loop_type, '\n', train.iloc[0].seqpair, '\n', train.iloc[0].rc)
def findprob(row):

    bppm=np.load(Path("../input/stanford-covid-vaccine/bpps") / f"{row.id}.npy")

    train.loc[train.id==row.id, 'pair_prob'] = pd.Series([np.sum(bppm, axis=0).tolist()] * len(train))

    ##the ugly hack again above. Somehow unable to get list into a cell without this hack



    pair=[]

    for i, structure in enumerate(row.structure):

        pair.append('1') if structure!='.' else pair.append('0')

    train.loc[train.id==row.id, 'paired'] = ''.join(pair)

    

train.apply(findprob, axis=1);
pd.set_option('display.max_colwidth', -1)

print('\n',train.iloc[0].sequence[:25], '\n', train.iloc[0].structure[:25], '\n', train.iloc[0].predicted_loop_type[:25], '\n\n', \

      train.iloc[0].seqpair[:25], '\n', train.iloc[0].rc[:25], '\n', train.iloc[0].paired[:25], '\n\n', train.iloc[0].pair_prob[:25])

pd.reset_option('display.max_colwidth')



plt.figure(figsize=(15,5))

plt.plot(train.iloc[0]['reactivity'])

plt.plot(train.iloc[0]['deg_Mg_50C'])
##convert sequences to list so it can explode

train['seqsplit']=[list(train.iloc[i]['sequence'][:68]) for i in range(len(train))]

train['seqpair']=[list(train.iloc[i]['seqpair'][:68]) for i in range(len(train))]

train['rc']=[list(train.iloc[i]['rc'][:68]) for i in range(len(train))]

train['paired']=[list(train.iloc[i]['paired'][:68]) for i in range(len(train))]

train['pair_prob']=[list(train.iloc[i]['pair_prob'][:68]) for i in range(len(train))]

train['seqpairpos']=[list(train.iloc[i]['seqpairpos'][:68]) for i in range(len(train))]

train['seqpairpos_dist']=[list(train.iloc[i]['seqpairpos_dist'][:68]) for i in range(len(train))]

##add an index col

train['numbering']=[np.arange(68).tolist() for i in range(len(train))]



##select all cols to be exploded

list_cols = ['seqsplit','numbering','seqpair','rc', 'paired','pair_prob','seqpairpos', 'seqpairpos_dist']

list_cols += train.filter(like="reactivity", axis=1).columns.tolist()

list_cols += train.filter(like="deg", axis=1).columns.tolist()

other_cols = list(set(train.columns) - set(list_cols))



exploded = [train[col].explode() for col in list_cols]

train_expanded = pd.DataFrame(dict(zip(list_cols, exploded)))

train_expanded = train[other_cols].merge(train_expanded, how="right", left_index=True, right_index=True).reset_index(drop=True)
int_cols = ['numbering','seqsplit','seqpair', 'seqpairpos', 'seqpairpos_dist','paired','pair_prob','reactivity','deg_Mg_pH10','deg_Mg_50C']

df=train_expanded[train_expanded.id=='id_001f94081'][int_cols]

display(df.transpose())
print('Mean Reactivity of paired base', round(df[df.seqpair!='X']['reactivity'].mean(),3))

print('Mean Reactivity of unpaired base', round(df[df.seqpair=='X']['reactivity'].mean(),3))

print('\nMean degradation(temp) of paired base', round(df[df.seqpair!='X']['deg_Mg_50C'].mean(),3))

print('Mean degradation(temp) of unpaired base', round(df[df.seqpair=='X']['deg_Mg_50C'].mean(),3))

print('\nMean degradation(Ph) of paired base', round(df[df.seqpair!='X']['deg_Mg_pH10'].mean(),3))

print('Mean degradation(Ph) of unpaired base', round(df[df.seqpair=='X']['deg_Mg_pH10'].mean(),3))
with pd.option_context('display.max_rows', None, 'display.max_columns', None):

    display(df[df.seqpair!='X'].transpose())
pd.reset_option('all')



from collections import Counter

rc_list=Counter((rc_dict)).most_common()

list(map(lambda x: print(x) if (len(x[0])>4) & (x[1]>8) else None, rc_list));
avg_reactivity = np.array(list(map(np.array,train.reactivity))).mean(axis=0)

avg_deg_Mg_50C = np.array(list(map(np.array,train.deg_Mg_50C))).mean(axis=0)

avg_deg_Mg_pH10 = np.array(list(map(np.array,train.deg_Mg_pH10))).mean(axis=0)



plt.figure(figsize=(18,6))

plt.plot(avg_reactivity)

plt.plot(avg_deg_Mg_50C)

plt.plot(avg_deg_Mg_pH10)



plt.show()
avg_deg_error_Mg_pH10 = np.array(list(map(np.array,train.deg_error_Mg_pH10))).mean(axis=0)

avg_deg_error_Mg_50C = np.array(list(map(np.array,train.deg_error_Mg_50C))).mean(axis=0)

avg_reactivity_error = np.array(list(map(np.array,train.reactivity_error))).mean(axis=0)



plt.figure(figsize=(18,6))

plt.plot(avg_reactivity_error)

plt.plot(avg_deg_error_Mg_50C)

plt.plot(avg_deg_error_Mg_pH10)



plt.show()