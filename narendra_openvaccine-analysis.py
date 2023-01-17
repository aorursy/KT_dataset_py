import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.express as px

import plotly.graph_objects as go





from pprint import pprint

from collections import defaultdict

from plotly.subplots import make_subplots
train_df=pd.read_json('../input/stanford-covid-vaccine/train.json', lines=True)

train_df=train_df[['id','seq_scored', 'seq_length', 'sequence', 'structure', 'predicted_loop_type','reactivity', 'deg_pH10', 'deg_Mg_pH10', 'deg_50C', 'deg_Mg_50C']].copy()

train_df.head()
base_2_idx={

    'A': 0,

    'U': 1,

    'G': 2,

    'C': 3

}



structure_2_idx={

    '(': 0,

    '.': 1,

    ')': 2

}



loop_idx= {

    'S': 0, 

    'M': 1, 

    'I': 2, 

    'B': 3, 

    'H': 4, 

    'E': 5, 

    'X':6

}
def getIndexes(row):

    sequence=row.sequence

    structure=row.structure

    predicted_loop_type=row.predicted_loop_type

    score_len=row.seq_scored

    

    nu_map=defaultdict(list)

    looptype_map=defaultdict(list)

    

    bp_list=[]

    non_bp_list=[]

    au_bp_list=[]

    gc_bp_list=[]

    

    

    for i in range(score_len):

        nucleotide=sequence[i]

        loop_type=predicted_loop_type[i]

        bp_structure=structure[i]

        

        nu_map[nucleotide].append(i)

        looptype_map[loop_type].append(i)

        

        if bp_structure == '.':

            non_bp_list.append(i)

        if bp_structure == ')' or bp_structure == '(':

            bp_list.append(i)

        if (bp_structure == ')' or bp_structure == '(') and (nucleotide == 'A' or nucleotide=='U'):

            au_bp_list.append(i)

        if (bp_structure == ')' or bp_structure == '(') and (nucleotide == 'G' or nucleotide=='C'):

            gc_bp_list.append(i)

    return {

        'nucleotide': nu_map,

        'loop_type': looptype_map,

        'bp_list': bp_list,

        'au_bp_list':au_bp_list,

        'gc_bp_list':gc_bp_list,

        'non_bp_list':non_bp_list

    }





def filter_by_indices(indices, filter_type, key):

    if filter_type == 'nucleotide':

        index=indices['nucleotide'][key]

    elif filter_type == 'loop_type':

        index=indices['loop_type'][key]

    else:

        index=indices[key]

    return index



def getDataSetByIndices(row, index_name):

    index=row[index_name]

    reactivity=[ val for i,val in enumerate(row.reactivity) if i in index]

    deg_Mg_pH10=[ val for i,val in enumerate(row.deg_Mg_pH10) if i in index]

    deg_Mg_50C=[ val for i,val in enumerate(row.deg_Mg_50C) if i in index]

    

    return pd.Series({

        'id': row.id,

        'reactivity': reactivity,

        'deg_Mg_pH10': deg_Mg_pH10,

        'deg_Mg_50C': deg_Mg_50C

    })
train_df['indices']=train_df.apply(getIndexes, axis=1)



for nucleotide in base_2_idx.keys():

    train_df[nucleotide+'_len']=train_df.indices.apply(lambda indices: len(indices['nucleotide'][nucleotide]))    

    train_df[nucleotide+"_index"]=train_df['indices'].apply(filter_by_indices,args=('nucleotide', nucleotide))

    

for loop_type in loop_idx.keys():

    train_df[loop_type+'_len']=train_df.indices.apply(lambda indices: len(indices['loop_type'][loop_type]))    

    train_df[loop_type+"_index"]=train_df['indices'].apply(filter_by_indices,args=('loop_type', loop_type))





train_df['bp_list_len']=train_df.indices.apply(lambda indices: len(indices['bp_list']))

train_df['bp_list_index']=train_df['indices'].apply(filter_by_indices,args=('bp_list', 'bp_list'))



train_df['au_bp_list_len']=train_df.indices.apply(lambda indices: len(indices['au_bp_list']))

train_df['au_bp_list_index']=train_df.indices.apply(filter_by_indices,args=('au_bp_list', 'au_bp_list'))



train_df['gc_bp_list_len']=train_df.indices.apply(lambda indices: len(indices['gc_bp_list']))

train_df['gc_bp_list_index']=train_df.indices.apply(filter_by_indices,args=('gc_bp_list', 'gc_bp_list'))





train_df['non_bp_list_len']=train_df.indices.apply(lambda indices: len(indices['non_bp_list']))

train_df['non_bp_list_index']=train_df.indices.apply(filter_by_indices,args=('non_bp_list', 'non_bp_list'))
loop_type_df={}

nucleotide_df={}



output_df=train_df[['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C']].copy()



bp_df=train_df.apply(getDataSetByIndices, axis=1, args=('bp_list_index', )).copy()

au_bp_df=train_df.apply(getDataSetByIndices, axis=1, args=('au_bp_list_index', )).copy()

gc_bp_df=train_df.apply(getDataSetByIndices, axis=1, args=('gc_bp_list_index', )).copy()

non_bp_df=train_df.apply(getDataSetByIndices, axis=1, args=('non_bp_list_index', )).copy()





for nucleotide in base_2_idx.keys():

    nucleotide_df[nucleotide]=train_df.apply(getDataSetByIndices, axis=1, args=(nucleotide+'_index', )).copy()



for loop_type in loop_idx.keys():

    loop_type_df[loop_type]=train_df.apply(getDataSetByIndices, axis=1, args=(loop_type+'_index', )).copy()



loop_type_df.keys()
def getColumnStats(df, colnames):

    for colname in colnames:

        df[colname+'_mean']=df[colname].apply(lambda seq: np.mean(seq) if len(seq)>0 else np.nan )

        df[colname+'_median'] =df[colname].apply(lambda seq: np.median(seq) if len(seq)>0 else np.nan )

        df[colname+'_std'] =df[colname].apply(lambda seq: np.std(seq) if len(seq)>0 else np.nan )

        df[colname+'_min'] =df[colname].apply(lambda seq: np.min(seq) if len(seq)>0 else np.nan )

        df[colname+'_max'] =df[colname].apply(lambda seq: np.max(seq) if len(seq)>0 else np.nan )
getColumnStats(output_df, ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C'])



getColumnStats(bp_df, ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C'])

getColumnStats(non_bp_df, ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C'])



getColumnStats(au_bp_df, ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C'])

getColumnStats(gc_bp_df, ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C'])



getColumnStats(non_bp_df, ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C'])
def countPlots(df, colnames, title):

    data={}

    for colname in colnames:

        data[colname]=df[colname].sum()

    fig=go.Figure(go.Bar(x=list(data.keys()), y=list(data.values())))

    fig.update_layout(title=title)

    fig.show()
nucleotide_colnames=['A_len','U_len','G_len','C_len']

structure_colnames=['bp_list_len','au_bp_list_len','gc_bp_list_len','non_bp_list_len']

loopTypeC_colnames=['S_len','M_len', 'I_len','B_len','H_len','E_len','X_len']





countPlots(train_df, nucleotide_colnames, 'Nucleotide')

countPlots(train_df, structure_colnames, 'Structure')

countPlots(train_df, loopTypeC_colnames, 'LoopTypes')
fig=make_subplots(rows=5, cols=3, shared_xaxes=True,

                  row_titles=['All', 'bp', 'AU-BP', 'GC-BP', 'Non-BP'],

                  column_titles=['Reactivity Mean', 'deg_Mg_pH10 Mean', 'deg_Mg_50C Mean'],

                  horizontal_spacing=0.1,

                  row_heights=[50, 50, 50, 50, 50]

                 )





fig.add_trace(go.Box(x=output_df.reactivity_mean),row=1,col=1)

fig.add_trace(go.Box(x=output_df.deg_Mg_pH10_mean), row=1, col=2)

fig.add_trace(go.Box(x=output_df.deg_Mg_50C_mean), row=1, col=3)





fig.add_trace(go.Box(x=bp_df.reactivity_mean),  row=2, col=1)

fig.add_trace(go.Box(x=bp_df.deg_Mg_pH10_mean), row=2, col=2)

fig.add_trace(go.Box(x=bp_df.deg_Mg_50C_mean),  row=2, col=3)





fig.add_trace(go.Box(x=au_bp_df.reactivity_mean),  row=3, col=1)

fig.add_trace(go.Box(x=au_bp_df.deg_Mg_pH10_mean), row=3, col=2)

fig.add_trace(go.Box(x=au_bp_df.deg_Mg_50C_mean),  row=3, col=3)





fig.add_trace(go.Box(x=gc_bp_df.reactivity_mean),  row=4, col=1)

fig.add_trace(go.Box(x=gc_bp_df.deg_Mg_pH10_mean), row=4, col=2)

fig.add_trace(go.Box(x=gc_bp_df.deg_Mg_50C_mean),  row=4, col=3)



fig.add_trace(go.Box(x=non_bp_df.reactivity_mean),  row=5, col=1)

fig.add_trace(go.Box(x=non_bp_df.deg_Mg_pH10_mean), row=5, col=2)

fig.add_trace(go.Box(x=non_bp_df.deg_Mg_50C_mean),  row=5, col=3)



fig.show()
fig=make_subplots(rows=5, cols=3, shared_xaxes=True,

                  row_titles=['All', 'bp', 'AU-BP', 'GC-BP', 'Non-BP'],

                  column_titles=['Reactivity Mean', 'deg_Mg_pH10 Mean', 'deg_Mg_50C Mean'],

                  horizontal_spacing=0.1,

                  row_heights=[50, 50, 50, 50, 50]

                 )





fig.add_trace(go.Histogram(x=output_df.reactivity_mean),row=1,col=1)

fig.add_trace(go.Histogram(x=output_df.deg_Mg_pH10_mean), row=1, col=2)

fig.add_trace(go.Histogram(x=output_df.deg_Mg_50C_mean), row=1, col=3)





fig.add_trace(go.Histogram(x=bp_df.reactivity_mean),  row=2, col=1)

fig.add_trace(go.Histogram(x=bp_df.deg_Mg_pH10_mean), row=2, col=2)

fig.add_trace(go.Histogram(x=bp_df.deg_Mg_50C_mean),  row=2, col=3)





fig.add_trace(go.Histogram(x=au_bp_df.reactivity_mean),  row=3, col=1)

fig.add_trace(go.Histogram(x=au_bp_df.deg_Mg_pH10_mean), row=3, col=2)

fig.add_trace(go.Histogram(x=au_bp_df.deg_Mg_50C_mean),  row=3, col=3)





fig.add_trace(go.Histogram(x=gc_bp_df.reactivity_mean),  row=4, col=1)

fig.add_trace(go.Histogram(x=gc_bp_df.deg_Mg_pH10_mean), row=4, col=2)

fig.add_trace(go.Histogram(x=gc_bp_df.deg_Mg_50C_mean),  row=4, col=3)



fig.add_trace(go.Histogram(x=non_bp_df.reactivity_mean),  row=5, col=1)

fig.add_trace(go.Histogram(x=non_bp_df.deg_Mg_pH10_mean), row=5, col=2)

fig.add_trace(go.Histogram(x=non_bp_df.deg_Mg_50C_mean),  row=5, col=3)



fig.show()
def getBasePairEncoding(structure):

    encodings=[]

    encid=0

    baseid=1



    for idx, ch in enumerate(structure):

        if ch == '(':

            encodings.append({

                'start_idx': idx

            })

            baseid+=1

        elif ch==')':

            baseid-=1

            encodings[encid]['end_idx']=idx

            encid+=1

    for encoding in encodings:

        encoding['dist']=encoding['end_idx']-encoding['start_idx']-1

    return encodings
train_df['bpEncoding'] = train_df.structure.apply(getBasePairEncoding)

train_df.head()
bpenc_df=train_df[['bpEncoding', 'reactivity', 'deg_Mg_pH10', 'deg_Mg_50C']].copy()

bpenc_df.head()
reactivity_diff=[]

ph10_diff=[]

mg_50c_diff=[]

dist=[]



def getResponseDiff(row):

    bpEncoding=row.bpEncoding

    reactivity=row.reactivity

    deg_Mg_pH10=row.deg_Mg_pH10

    deg_Mg_50C=row.deg_Mg_50C

    

    reactivity_len=len(reactivity)

    for enc in bpEncoding:

        start_idx=enc['start_idx']

        end_idx=enc['end_idx']

        d=enc['dist']

        

        if start_idx>=reactivity_len or end_idx>=reactivity_len:

            break

        reactivity_diff.append( abs(reactivity[start_idx]-reactivity[end_idx]) )

        ph10_diff.append( abs(deg_Mg_pH10[start_idx] - deg_Mg_pH10[end_idx]) )

        mg_50c_diff.append( abs(deg_Mg_50C[start_idx] - deg_Mg_50C[end_idx]) )

        dist.append(d)

_=bpenc_df.apply(getResponseDiff, axis=1)
fig=make_subplots(rows=2, cols=3)



fig.add_trace(go.Box(x=reactivity_diff),row=1,col=1)

fig.add_trace(go.Box(x=ph10_diff),row=1,col=2)

fig.add_trace(go.Box(x=mg_50c_diff),row=1,col=3)







fig.add_trace(go.Histogram(x=reactivity_diff),row=2,col=1)

fig.add_trace(go.Histogram(x=ph10_diff), row=2, col=2)

fig.add_trace(go.Histogram(x=mg_50c_diff), row=2, col=3)
