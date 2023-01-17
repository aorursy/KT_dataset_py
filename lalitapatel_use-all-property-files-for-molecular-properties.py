from IPython.core.display import display, HTML, Javascript

import IPython.display



html_string = """

<g id="colimg"></g>

"""

js_string = """

require.config({paths:{d3: "https://d3js.org/d3.v4.min"}});

require(["d3"], function(d3) {d3.select("#colimg").append("img").attr("src", "http://lipy.us/img/Columns.png");});

"""

h = display(HTML(html_string))

j = IPython.display.Javascript(js_string)

IPython.display.display_javascript(j)
%%capture

'''

import numpy as np

import pandas as pd

import time

import gc

'''
%%capture

'''

def reduceMemory(df):

    

    beg_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in ['int16','int32','int64','float16','float32','float64']:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min >= np.iinfo(np.int64).min and c_max <= np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min >= np.finfo(np.float16).min and c_max <= np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage reduced by {0:0.1f} % to {0:5.2f} Mb'.format(100*(beg_mem-end_mem)/(beg_mem), end_mem))

        

    return df

'''
%%capture

'''

df0 = pd.read_csv('../input/structures.csv')

'''
%%capture

'''

df0 = pd.merge(df0, pd.DataFrame([['H',0.43,2.2], ['C',0.82,2.55], ['N',0.80,3.04], ['O',0.78,3.44], ['F',0.76,3.98]],

                    columns=['atom','rad','EN']), how='left', on=['atom']) # radius, electronegativity

'''
%%capture

'''

ind1 = df0['atom_index'].values



mol1 = df0['molecule_name'].values

pos1 = df0[['x', 'y', 'z']].values

rad1 = df0['rad'].values



mol2 = mol1

pos2 = pos1

rad2 = rad1



atmx = 28

dlen = len(df0)

rec1 = np.arange(dlen)

bond = np.zeros((dlen+1, atmx+1), dtype=np.int8)

bdis = np.zeros((dlen+1, atmx+1), dtype=np.float32)



for atmi in range(atmx-1):

    

    mol2 = np.roll(mol2, -1, axis=0)

    pos2 = np.roll(pos2, -1, axis=0)

    rad2 = np.roll(rad2, -1, axis=0)

  

    mask = np.where(mol1==mol2, 1, 0)

    dist = np.linalg.norm(pos1 - pos2, axis=1) * mask

    chec = np.where(np.logical_and(dist > 0.0001, dist < rad1 + rad2), 1, 0)  

    

    ind1 = ind1

    ind2 = ind1 + atmi + 1

    ind2 = np.where(np.logical_or(ind2 > atmx, mask==0), atmx, ind2)

    

    rec1 = rec1

    rec2 = rec1 + atmi + 1

    rec2 = np.where(np.logical_or(rec2 > dlen, mask==0), dlen, rec2)



    bond[(rec1, ind2)] = chec

    bond[(rec2, ind1)] = chec

    bdis[(rec1, ind2)] = dist

    bdis[(rec2, ind1)] = dist



bond = np.delete(bond, axis=0, obj=-1) # Delete dummy row.

bond = np.delete(bond, axis=1, obj=-1) # Delete dummy col.



bdis = np.delete(bdis, axis=0, obj=-1) # Delete dummy row.

bdis = np.delete(bdis, axis=1, obj=-1) # Delete dummy col.



bnum = [[ i for i,x in enumerate(row) if x] for row in bond ]

bqty = [ len(x) for x in bnum ]

blen = [[ dist for i,dist in enumerate(row) if i in bnum[j] ] for j,row in enumerate(bdis)]



blen_avg = [ np.mean(x) for x in blen ]

blen_med = [ np.median(x) for x in blen ]

blen_std = [ np.std(x) for x in blen ]



df0 = df0.join(pd.DataFrame({'bond_num':bqty, 'bondleng_avg':blen_avg, 'bondleng_med':blen_med, 'bondleng_std':blen_std}))

'''
%%capture

'''

df0 = pd.merge(df0, pd.read_csv('../input/dipole_moments.csv'), how='left', on=['molecule_name'])

'''
%%capture

'''

df0 = pd.merge(df0, pd.read_csv('../input/potential_energy.csv'), how='left', on=['molecule_name'])

'''
%%capture

'''

df0 = pd.merge(df0, pd.read_csv('../input/mulliken_charges.csv'), how='left', on=['molecule_name','atom_index'])

'''
%%capture

'''

df0 = pd.merge(df0, pd.read_csv('../input/magnetic_shielding_tensors.csv'), how='left', on=['molecule_name','atom_index'])

'''
%%capture

'''

df0 = reduceMemory(df0)

'''
%%capture

'''

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

'''
%%capture

'''

train = reduceMemory(train)

test = reduceMemory(test)

'''
%%capture

'''

train.head()

'''
%%capture

'''

df1 = pd.read_csv('../input/scalar_coupling_contributions.csv')

train = pd.merge(train, df1, how='left', on=['molecule_name','atom_index_0','atom_index_1','type'])

test = pd.merge(test, df1, how='left', on=['molecule_name','atom_index_0','atom_index_1','type'])

'''
%%capture

'''

df0.columns = [(c if (c=="molecule_name") else c+"_0") for c in df0.columns]

train = pd.merge(train, df0, how='left', on=['molecule_name','atom_index_0'])

test = pd.merge(test, df0, how='left', on=['molecule_name','atom_index_0'])

df0.columns = [(c if (c=="molecule_name") else c.replace("_0","_1")) for c in df0.columns]

train = pd.merge(train, df0, how='left', on=['molecule_name','atom_index_1'])

test = pd.merge(test, df0, how='left', on=['molecule_name','atom_index_1'])

'''
%%capture

'''

train = reduceMemory(train)

test = reduceMemory(test)

'''
%%capture

'''

def addStat(df):

    

    df['dist'] = np.linalg.norm( df[['x_0','y_0','z_0']].values - df[['x_1','y_1','z_1']].values, axis=1 )

    df['dist'] = 1/(df['dist']**3)

    for w in ['x','y','z']:

        df['dist_'+w] = (df[w+'_0'] - df[w+'_1']) ** 2

    df['type_0'] = df['type'].apply(lambda x: x[0])

    

    df['molecule_couples'] = df.groupby('molecule_name')['id'].transform('count')

    df['atom_index_0_couples'] = df.groupby(['molecule_name','atom_index_0'])['id'].transform('count')

    df['atom_index_1_couples'] = df.groupby(['molecule_name','atom_index_1'])['id'].transform('count')

    

    for zact in ['min','max','mean','std']:

        df['molecule_dist_'+zact] = df.groupby('molecule_name')['dist'].transform(zact)

        

        for zfor in ['dist']: # ['x_1','y_1','z_1','dist']:

            for zato in ['atom_index_0','atom_index_1','type','atom_0','atom_1','type_0']:

                try:

                    df[f'molecule_'+zato+'_'+zfor+'_'+zact] = df.groupby(['molecule_name',zato])[zfor].transform(zact)

                except:

                    df[f'molecule_'+zato+'_'+zfor+'_'+zact] = 0

                try:

                    df[f'molecule_'+zato+'_'+zfor+'_'+zact+'_dif'] = df[f'molecule_'+zato+'_'+zfor+'_'+zact] - df[zfor]

                except:

                    df[f'molecule_'+zato+'_'+zfor+'_'+zact+'_dif'] = 0

                try:

                    df[f'molecule_'+zato+'_'+zfor+'_'+zact+'_div'] = df[f'molecule_'+zato+'_'+zfor+'_'+zact] / df[zfor]

                except:

                    df[f'molecule_'+zato+'_'+zfor+'_'+zact+'_div'] = 0

              

    return df

'''
%%capture

'''

train = addStat(train)

test = addStat(test)

'''
%%capture

'''

print("Dataset                   Rows   Columns")

print('{0:20s}{1:10d}{2:10d}'.format("df0", df0.shape[0], df0.shape[1]))

print('{0:20s}{1:10d}{2:10d}'.format("df1", df1.shape[0], df1.shape[1]))

print('{0:20s}{1:10d}{2:10d}'.format("train", train.shape[0], train.shape[1]))

print('{0:20s}{1:10d}{2:10d}'.format("test", test.shape[0], test.shape[1]))

'''
%%capture

'''

train.to_csv('train_plus.csv', index=True)

test.to_csv('test_plus.csv', index=True)

'''
# This is to register this kernel in this specific challenge.

import pandas as pd

pd.read_csv('../input/sample_submission.csv')[:10]