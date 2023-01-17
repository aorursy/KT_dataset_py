import numpy as np 
import pandas as pd 

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
difficulty_level = 3
train = train[train['difficulty'] == difficulty_level]
test = test[test['difficulty'] == difficulty_level]
cipher_path = '../input/cipher-1-cipher-2-full-solutions/'
cipher2_map = pd.read_csv(cipher_path + '/cipher2_map.csv')
translation_2 = str.maketrans(''.join(cipher2_map['cipher']), ''.join(cipher2_map['plain']))
train['t_text'] = train.apply(lambda x: x['text'].translate(translation_2), axis=1)
df_p_extract = df_p[df_p['text'].str.startswith('From:')].copy()
df_p_extract['text'] = df_p_extract['text'].map(lambda x: x[:300])
df_p_extract['p_len'] = df_p_extract['text'].map(len)
df_p_list = []
for i in np.sort(df_p_extract['target'].unique()):
    df_p_list.append(df_p_extract[df_p_extract['target'] == i])
df_c = train[train['t_text'].str.startswith('FrMmZ')].copy()
len(df_c)
def find_match(idx):
    target = df_c.loc[idx,'target']
    t_text = df_c.loc[idx,'t_text']
    t_len = len(t_text)
    df_p_match = df_p_list[target][df_p_list[target]['p_len']==t_len]
    p_text, fscore, p_index =  process.extractOne(t_text, df_p_match['text'], scorer = fuzz.token_set_ratio)
    return(p_text, fscore, p_index)
par_compute = [delayed(find_match)(idx) for idx in df_c.index]
cp_matches = compute(*par_compute, scheduler='processes')
cp_matches = pd.DataFrame(list(cp_matches),columns=['p_text','fscore','p_index'])
df_c = df_c[['target','text','t_text']].reset_index().rename(columns={'index' : 'c_index', 'text' : 'c_text'})
df_c = pd.concat([df_c,cp_matches],axis=1)
df_c.sort_values(by='fscore',ascending=False,inplace=True)
df_c.head()
df_c_copy = df_c.copy()
def word_freqs(s, seps):
    words = list(filter(None, re.split('[' + ''.join(seps) + ']+',s)))
    freqs = pd.Series(words).value_counts()
    freqs = freqs.reset_index().rename(columns={'index' : 'word', 0:'count'})
    freqs['word_len'] = freqs['word'].map(len)
    freqs['abs_freq'] = 100 * freqs['count'] / len(words)
    freqs = pd.merge(freqs,
                     freqs.groupby('word_len')[['count']].sum().reset_index().rename(columns={'count' : 'word_len_count'}),
                     on='word_len')
    freqs['rel_freq'] = 100 * freqs['count'] / freqs['word_len_count']
    freqs.sort_values(by='abs_freq',ascending=False,inplace=True)
    return(freqs)
plaintext = ' '.join(df_p_extract['text'])
p_words = word_freqs(plaintext,[' '])
p_words.head()
words = ['Subject:','Organization','Lines:'] 
t_words = [r'\s*(..bje.t.)\s', r'\s*(Or..n...t..n.)\s', r'\s*(..nes.)\s']
#The above regular expressions have been inferred by manually looking at a few ciphertexts
for i, t in enumerate(t_words):
    w = words[i]
    df_c[w + '_is'] = df_c['p_text'].map(lambda x: [match.span()[0] for match in re.finditer(w, x) if match is not None])
    df_c[w + '_is_t'] = df_c['t_text'].map(lambda x: [match.span(1)[0] for match in re.finditer(t, x,re.DOTALL) if match is not None])
df_c.head()
def frequent_word_match(x):
    res = True
    for i, t in enumerate(t_words):
        w = words[i]
        res = res and (x[w + '_is'] == x[w + '_is_t'])
    return(res)
df_c['freq_word_match'] = df_c.apply(lambda x: frequent_word_match(x),axis = 1)
len(df_c[~df_c['freq_word_match']])
df_crib = df_c[df_c['freq_word_match']].copy()
len(df_crib)
def word_aligned(x):
    t_text = x['t_text']
    p_text = x['p_text']
    t_list = t_text.split(' ')
    p_list = p_text.split(' ')
    return [len(s) for s in t_list] == [len(s) for s in p_list]
df_crib['word_aligned'] = df_crib.apply(lambda x: word_aligned(x),axis=1)
df_crib_misaligned = df_crib[~df_crib['word_aligned']]
len(df_crib_misaligned)
#We may investigate these misaligned cipher & plaintext pairs later
df_crib = df_crib[df_crib['word_aligned']]
len(df_crib)
df_crib = df_crib[['target','c_index','c_text','p_text','p_index']]
df_crib.head()
translation_2_ct = str.maketrans(''.join(cipher2_map['cipher']), ''.join(cipher2_map['plain'])) # cipher #2 decryption
translation_2_pt = str.maketrans(''.join(cipher2_map['plain']),''.join(cipher2_map['cipher'])) # cipher #2 encryption
# Checking that no characters are missing in cipher #2 map to encrypt the cipher #3 plaintexts from the crib

cipher2_plain_alphabet = set(''.join(cipher2_map['plain']))
df_crib['p_text_ok'] = df_crib['p_text'].map(lambda x: len(set(x).difference(cipher2_plain_alphabet)) == 0)
len(df_crib[~df_crib['p_text_ok']])
df_crib.drop('p_text_ok',axis=1,inplace=True)
df_crib['pt_text'] = df_crib['p_text'].map(lambda x: x.translate(translation_2_pt))
df_crib['ct_text'] = df_crib['c_text'].map(lambda x: x.translate(translation_2_ct))
df_crib.to_pickle('df_crib.pkl')
def compare_ptc(idx):

    p_text = df_crib['p_text'].loc[idx]
    ct_text = df_crib['ct_text'].loc[idx]
    
    pt_text = df_crib['pt_text'].loc[idx]
    c_text = df_crib['c_text'].loc[idx]
    
    c_split = c_text.split('8')
    pt_split = pt_text.split('8')
    ct_split = ct_text.split(' ')
    p_split = p_text.split(' ')

    return(pd.DataFrame([p_split,ct_split,pt_split, c_split],index=['p','ct','pt','c']).T)    
def hide_ok_nok(x,pt = True, hide_ok = True):
    pt_w = x['pt']
    c_w = x['c']
    if pt:
        res = pt_w
    else:
        res = c_w
    ok_i = set([i for i,(a,b) in enumerate(zip(pt_w,c_w)) if (ord(a) ^ ord(b) == 0)])
    if hide_ok:
        return(''.join(['.' if i in ok_i else res[i] for i in range(len(pt_w))]))
    else:
        return(''.join(['.' if i not in ok_i else res[i] for i in range(len(pt_w))]))
df_crib.head(2)
df_z = compare_ptc(846)
display(df_z.applymap(repr).T)

df_z['pt_h_hide_ok'] = df_z.apply(lambda x: hide_ok_nok(x,pt=True),axis=1)
df_z['c_hide_ok'] = df_z.apply(lambda x: hide_ok_nok(x,pt=False),axis=1)
df_z['pt_h_hide_nok'] = df_z.apply(lambda x: hide_ok_nok(x,pt=True,hide_ok=False),axis=1)
df_z['c_hide_nok'] = df_z.apply(lambda x: hide_ok_nok(x,pt=False,hide_ok=False),axis=1)
display(df_z.applymap(repr).T)

df_zz = pd.DataFrame([list(''.join(df_z['pt_h_hide_ok'])),list(''.join(df_z['c_hide_ok'])),list(''.join(df_z['pt_h_hide_nok'])),list(''.join(df_z['c_hide_nok']))],index=['pt_h_hide_ok','c_hide_ok','pt_h_hide_nok','c_hide_nok'])
display(df_zz)

pt_h = ''.join(df_zz.loc['pt_h_hide_ok'])
print('Characters to further encipher')
print(repr(pt_h))
pt_h_n = ''.join(df_zz.loc['pt_h_hide_nok'])
print('Characters of cipher#2 equal to cipher#3')
print(repr(pt_h_n))
df_z = compare_ptc(549)
display(df_z.applymap(repr).T)

df_z['pt_h_hide_ok'] = df_z.apply(lambda x: hide_ok_nok(x,pt=True),axis=1)
df_z['c_hide_ok'] = df_z.apply(lambda x: hide_ok_nok(x,pt=False),axis=1)
df_z['pt_h_hide_nok'] = df_z.apply(lambda x: hide_ok_nok(x,pt=True,hide_ok=False),axis=1)
df_z['c_hide_nok'] = df_z.apply(lambda x: hide_ok_nok(x,pt=False,hide_ok=False),axis=1)
display(df_z.applymap(repr).T)

df_zz = pd.DataFrame([list(''.join(df_z['pt_h_hide_ok'])),list(''.join(df_z['c_hide_ok'])),list(''.join(df_z['pt_h_hide_nok'])),list(''.join(df_z['c_hide_nok']))],index=['pt_h_hide_ok','c_hide_ok','pt_h_hide_nok','c_hide_nok'])
display(df_zz)

pt_h = ''.join(df_zz.loc['pt_h_hide_ok'])
print('Characters to further encipher')
print(repr(pt_h))
pt_h_n = ''.join(df_zz.loc['pt_h_hide_nok'])
print('Characters of cipher#2 equal to cipher#3')
print(repr(pt_h_n))
df_z = compare_ptc(846)
pt_t = re.compile(r'[\W\d_]+').sub('', ''.join(df_z['pt']))
c_t = re.compile(r'[\W\d_]+').sub('', ''.join(df_z['c']))
df_3 = pd.DataFrame([list(pt_t),list(c_t)],index=['t','c'])
df_3
df_3_n = df_3.applymap(ord)
df_3_n = pd.concat([df_3_n,pd.DataFrame(df_3_n.loc['c'] - df_3_n.loc['t'],columns=['diff']).T],axis=0)
df_3_n = pd.concat([df_3_n,pd.DataFrame(df_3_n.loc['diff'].map(lambda x: x + 26 if (x <-1) else x)).rename(columns={'diff' : 'diffMod26'}).T],axis=0)
df_3_n
key_ord = [7, 4, 11, 4, 13, -1, 5, 14, 20, 2, 7, 4, -1, 6, 0, 8, 13, 4, 18]
df_3_n = pd.concat([df_3_n,pd.DataFrame(list(islice(cycle(key_ord), len(df_3_n.columns))),columns=['key']).T],axis=0)
df_3_n
(df_3_n.loc['diffMod26'] - df_3_n.loc['key']).map(abs).sum()
df_z = compare_ptc(549)
pt_t = re.compile(r'[\W\d_]+').sub('', ''.join(df_z['pt']))
c_t = re.compile(r'[\W\d_]+').sub('', ''.join(df_z['c']))
df_3 = pd.DataFrame([list(pt_t),list(c_t)],index=['t','c'])
df_3
df_3_n = df_3.applymap(ord)
df_3_n = pd.concat([df_3_n,pd.DataFrame(df_3_n.loc['c'] - df_3_n.loc['t'],columns=['diff']).T],axis=0)
df_3_n = pd.concat([df_3_n,pd.DataFrame(df_3_n.loc['diff'].map(lambda x: x + 26 if (x <-1) else x)).rename(columns={'diff' : 'diffMod26'}).T],axis=0)
df_3_n
df_3_n = pd.concat([df_3_n,pd.DataFrame(list(islice(cycle(key_ord), len(df_3_n.columns))),columns=['key']).T],axis=0)
(df_3_n.loc['diffMod26'] - df_3_n.loc['key']).map(abs).sum()
key_char = [chr(i+ord('a')) if i>=0 else ' ' for i in key_ord]
''.join(key_char)
train.drop('t_text',axis=1,inplace=True)
train.head()
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
    l_alpha_shifted = [shift_char(c,-shift) for c, shift in zip(l_alpha,list(islice(cycle(key_ord), len(l_alpha))))]
    return(''.join(replace_alpha(l,l_alpha_shifted)))
train['ct_text'] = train['text'].map(lambda x: fractional_vigenere(x,key_ord).translate(translation_2_ct))
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
train.to_pickle('train_3.pkl')
test['ct_text'] = test['text'].map(lambda x: fractional_vigenere(x,key_ord).translate(translation_2_ct))
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
test.to_pickle('test_3.pkl')