import numpy as np 
import pandas as pd 

import itertools
import os
import re

from collections import Counter
from dask import delayed, compute
from dask.diagnostics import ProgressBar
from fuzzywuzzy import fuzz, process
from IPython.core.display import display
from itertools import cycle, islice
from sklearn.datasets import fetch_20newsgroups

ProgressBar().register()

chunk_size = 300
pd.options.display.max_columns = chunk_size
pd.options.display.max_rows = chunk_size
train_p = fetch_20newsgroups(subset='train')
test_p = fetch_20newsgroups(subset='test')
df_p = pd.concat([pd.DataFrame(data = np.c_[train_p['data'], train_p['target']],
                                   columns= ['text','target']),
                      pd.DataFrame(data = np.c_[test_p['data'], test_p['target']],
                                   columns= ['text','target'])],
                     axis=0).reset_index(drop=True)
df_p['target'] = df_p['target'].astype(np.int8)
df_p['text'] = df_p['text'].map(lambda x: x.replace('\r\n','\n').replace('\r','\n').replace('\n','\n '))
df_p.loc[df_p['text'].str.endswith('\n '),'text'] = df_p.loc[df_p['text'].str.endswith('\n '),'text'].map(lambda x: x[:-1])
p_text_chunk_list = []
p_text_index_list = []

for p_index, p_row in df_p.iterrows():
    p_text = p_row['text']
    p_text_len = len(p_text)
    if p_text_len > chunk_size:
        for j in range(p_text_len // chunk_size):
            p_text_chunk_list.append(p_text[chunk_size*j:chunk_size*(j+1)])
            p_text_index_list.append(p_index)
        if p_text_len%chunk_size > 0:
            p_text_chunk_list.append(p_text[chunk_size*(p_text_len // chunk_size):(chunk_size*(p_text_len // chunk_size)+p_text_len%chunk_size)])
            p_text_index_list.append(p_index)
    else:
        p_text_chunk_list.append(p_text)
        p_text_index_list.append(p_index)
df_p_chunked = pd.DataFrame({'text' : p_text_chunk_list, 'p_index' : p_text_index_list})
df_p_chunked = pd.merge(df_p_chunked, df_p.reset_index().rename(columns={'index' : 'p_index'})[['p_index','target']],on='p_index',how='left')

df_p_chunked_list = []
for i in np.sort(df_p_chunked['target'].unique()):
    df_p_chunked_list.append(df_p_chunked[df_p_chunked['target'] == i])
competition_path = '../input/20-newsgroups-ciphertext-challenge/'
train = pd.read_csv(competition_path + 'train.csv').rename(columns={'ciphertext' : 'text'})
test = pd.read_csv(competition_path + 'test.csv').rename(columns={'ciphertext' : 'text'})
cipher_path = '../input/cipher-1-cipher-2-full-solutions/'

cipher2_map = pd.read_csv(cipher_path + '/cipher2_map.csv')

cipher2_map = pd.concat([cipher2_map,pd.DataFrame(data=[['D','\x10']],columns=['cipher','plain'])],axis=0,ignore_index=True)
#Cheating a bit with this update, the cipher character D has to be added to the cipher_2 map because a cipher #4 text, it would have appeared as missing in our decryption matching at the end of this kernel

translation_2_pt = str.maketrans(''.join(cipher2_map['plain']),''.join(cipher2_map['cipher'])) # cipher #2 encryption
translation_2_ct = str.maketrans(''.join(cipher2_map['cipher']), ''.join(cipher2_map['plain'])) # cipher #2 decryption

def shift_char(c,shift):
    if c.islower():
        return(chr((ord(c) - ord('a') + shift) % 26 + ord('a')))
    else:
        return(chr((ord(c) - ord('A') + shift) % 26 + ord('A')))

def replace_alpha(l,l_alpha_new):
    res = []
    i_alpha = 0
    for i in range(len(l)):
        if l[i].isalpha():
            res.append(l_alpha_new[i_alpha])
            i_alpha += 1
        else:
            res.append(l[i])
    return(res)

def fractional_vigenere(s,key):
    l = list(s)
    l_alpha = [x for x in l if x.isalpha()]
    l_alpha_shifted = [shift_char(c,-shift) for c, shift in zip(l_alpha,list(islice(cycle(key), len(l_alpha))))]
    return(''.join(replace_alpha(l,l_alpha_shifted)))

key_ord_3 = [7, 4, 11, 4, 13, -1, 5, 14, 20, 2, 7, 4, -1, 6, 0, 8, 13, 4, 18] # cipher #3 decryption key
key_ord_3_n = [-x for x in key_ord_3] # cipher #3 encryption key
p_counts = pd.Series(Counter(''.join(df_p['text']))).rename("counts").to_frame().sort_values("counts", ascending = False)
p_counts = 1000000 * p_counts / p_counts.sum()
p_counts = p_counts.reset_index().rename(columns = {"index":"p_char"})

c_counts = []
for i in range(1,5):
    counts = pd.Series(Counter(''.join(pd.concat([train[train['difficulty'] == i][['text']],test[test['difficulty'] == i][['text']]],axis=0)['text']))).rename('counts').to_frame().sort_values('counts', ascending = False)
    counts = 1000000 * counts / counts.sum()
    counts = counts.reset_index().rename(columns = {'index':'c_{}_char'.format(i)})
    c_counts.append(counts)
pd.concat([p_counts] + c_counts, axis = 1).head(20)
df_p_chunked['text_len'] = df_p_chunked['text'].map(len) 
df_p_chunked['pt_text'] = df_p_chunked['text'].map(lambda x: fractional_vigenere(x.translate(translation_2_pt),key_ord_3_n))
df_p_chunked['pt_counter'] = df_p_chunked['pt_text'].map(lambda x: Counter(x))
df_p_chunked = df_p_chunked.reset_index().rename(columns={'index' : 'p_chunk_index'})
difficulty_level = 4

train = train[train['difficulty'] == difficulty_level]
train['text_len'] = train['text'].map(len) 

test = test[test['difficulty'] == difficulty_level]
res = []
for i in train.index[:]:
    c_text = train.loc[i]['text']
    c_target = train.loc[i]['target']
    c_len = train.loc[i]['text_len']
    c_counter = Counter(c_text)
    p_chunk_indexes = list(df_p_chunked[(df_p_chunked['text_len'] == c_len) & (df_p_chunked['target'] == c_target) & (df_p_chunked['pt_counter'] == c_counter)]['p_chunk_index'].values)
    if len(p_chunk_indexes) == 1:
        res.append((i,p_chunk_indexes[0]))
df_crib2 = pd.DataFrame(res,columns=['train_index','p_chunk_index'])
df_crib2['c_text'] = df_crib2['train_index'].map(lambda x: train.loc[x,'text'])
df_crib2['pt_text'] = df_crib2['p_chunk_index'].map(lambda x: df_p_chunked.loc[x,'pt_text'])
df_crib2['text_len'] = df_crib2['c_text'].map(len)
def trans_mapper(x):
    pt_text = x['pt_text']
    c_text = x['c_text']
    
    pt_series = pd.Series(Counter(pt_text)).rename('pt_counts').to_frame().sort_values('pt_counts', ascending = False)
    c_series = pd.Series(Counter(c_text)).rename('c_counts').to_frame().sort_values('c_counts', ascending = False)
    ptc_series = pd.merge(pt_series, c_series, left_index=True,right_index=True,how='outer')

    if len(ptc_series[ptc_series['pt_counts'] != ptc_series['c_counts']]) > 0:
        return(np.nan)

    trans_map = ptc_series[ptc_series['pt_counts'] == 1]
    trans_map = trans_map.reset_index().rename(columns={'index' : 'char'})
    trans_map['p_char_index'] = trans_map['char'].map(lambda x: pt_text.find(x))
    trans_map['c_char_index'] = trans_map['char'].map(lambda x: c_text.find(x))

    return(dict(zip(trans_map['p_char_index'].values,trans_map['c_char_index'].values)))
df_crib2['trans_map'] = df_crib2.apply(lambda x: trans_mapper(x),axis=1)
trans_len = 300
trans_dict = {}
for i in range(trans_len):
    temp = set()
    for j in df_crib2[df_crib2['text_len'] == trans_len].index:
        t_map = df_crib2.loc[j,'trans_map']
        if i in t_map:
            temp = temp.union([t_map[i]])
    trans_dict[i] = temp
trans_s = pd.Series(trans_dict)
print(trans_s.map(len).min())
print(trans_s.map(len).max())
trans_s = trans_s.map(lambda x: list(x)[0] if len(list(x))>0 else np.nan)
def rowcol(i,n_cols):
    row = i // n_cols
    col = i % n_cols
    return((row,col))

def draw_square(n,n_cols):
    n_rows = rowcol(n,n_cols)[0]
    df = pd.DataFrame(data=[[-1]*n_cols]*n_rows,index=range(n_rows),columns=range(n_cols))
    for i in range(n):
        df.loc[rowcol(i,n_cols)] = i
    return(df)
trans_s.rename('c_char_index').to_frame().reset_index().rename(columns={'index':'p_char_index'}).sort_values(by='c_char_index').T.style.format("{:.0f}")
draw_square(300,24).style.format("{:.0f}")
def read_col(col,square):
    n_rows = len(square)
    res = []
    for i in range(n_rows):
        x_col = square.loc[i,col]
        if ~np.isnan(x_col):
            res = res + [x_col]
    return(res)

def alternate_cols(col_1,col_2,square):
    n_rows = len(square)
    res = []
    for i in range(n_rows):
        x_col1 = square.loc[i,col_1]
        x_col2 = square.loc[i,col_2]
        if ~np.isnan(x_col1):
            res = res + [x_col1]
        if ~np.isnan(x_col2):
            res = res + [x_col2]
    return(res)
encipher_trans_dict = {}
decipher_trans_dict = {}
for i in range(1,301):
    df = draw_square(i,24)
    res = []
    res = res + read_col(0,df)
    for j in range(1,12):
        res = res + alternate_cols(j,24-j,df)
    res = res + read_col(12,df)
    encipher_trans_dict[i] = res
    decipher_trans_dict[i] = np.argsort(res)
def encipher4(p_text):
    p_len = len(p_text)
    res = encipher_trans_dict[p_len]
    return(''.join([p_text[int(res[i])] for i in range(p_len)]))

def decipher4(c_text):
    c_len = len(c_text)
    res = decipher_trans_dict[c_len]
    return(''.join([c_text[int(res[i])] for i in range(c_len)]))
train.head()
train['ct_text'] = train['text'].map(lambda x: fractional_vigenere(decipher4(x),key_ord_3).translate(translation_2_ct))
target_list = np.sort(df_p_chunked['target'].unique())
p_indexes_dict = {}
for i in target_list[:]:
    df = df_p_chunked_list[i]
    for j in train[train['target'] == i].index[:]:
        ct_text = train.loc[j,'ct_text']
        new_p_indexes = set(df[df['text'] == ct_text]['p_index'])
        if len(new_p_indexes) > 0:
            p_indexes_dict[j] = p_indexes_dict.get(j,set()).union(new_p_indexes)
train_p_indexes = pd.DataFrame(pd.Series(data=list(p_indexes_dict.values()), index = p_indexes_dict.keys(),dtype=object)).rename(columns={0:'p_indexes'})
print(train.shape[0])
print(train_p_indexes.shape[0])
train = train.join(train_p_indexes)
train.to_pickle('train_4.pkl')
test['ct_text'] = test['text'].map(lambda x: fractional_vigenere(decipher4(x),key_ord_3).translate(translation_2_ct))
p_indexes_dict = {}
for i in target_list[:]:
    df = df_p_chunked_list[i]
    for j in test.index[:]:
        t_text = test.loc[j,'ct_text']
        new_p_indexes = set(df[df['text'] == ct_text]['p_index'])
        if len(new_p_indexes) > 0:
            p_indexes_dict[j] = p_indexes_dict.get(j,set()).union(new_p_indexes)
test_p_indexes = pd.DataFrame(pd.Series(data=list(p_indexes_dict.values()), index = p_indexes_dict.keys(),dtype=object)).rename(columns={0:'p_indexes'})
print(test.shape[0])
print(test_p_indexes.shape[0])
test = test.join(test_p_indexes)
test.to_pickle('test_4.pkl')