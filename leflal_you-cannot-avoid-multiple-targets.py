import numpy as np 
import pandas as pd 

import os

from IPython.core.display import display
from sklearn.datasets import fetch_20newsgroups
print(os.listdir("../input"))
competition_path = '20-newsgroups-ciphertext-challenge'
test = pd.read_csv('../input/' + competition_path + '/test.csv').rename(columns={'ciphertext' : 'text'})
train_p = fetch_20newsgroups(subset='train')
test_p = fetch_20newsgroups(subset='test')
df_p = pd.concat([pd.DataFrame(data = np.c_[train_p['data'], train_p['target']],
                                   columns= ['text','target']),
                      pd.DataFrame(data = np.c_[test_p['data'], test_p['target']],
                                   columns= ['text','target'])],
                     axis=0).reset_index(drop=True)
df_p['text'] = df_p['text'].map(lambda x: x.replace('\r\n','\n').replace('\r','\n').replace('\n','\n '))
df_p.loc[df_p['text'].str.endswith('\n '),'text'] = df_p.loc[df_p['text'].str.endswith('\n '),'text'].map(lambda x: x[:-1])
df_p['target'] = df_p['target'].astype(np.int8)
cipher_path = 'cipher-1-cipher-2-full-solutions'
cipher1_map = pd.read_csv('../input/'+ cipher_path + '/cipher1_map.csv')
translation_1 = str.maketrans(''.join(cipher1_map['cipher']), ''.join(cipher1_map['plain']))
test.loc[40]
c_text = test.loc[40,'text']
t_text = test.loc[40,'text'].translate(translation_1)
t_text
df_p.loc[[4473,7227],'text'].str.contains(t_text,regex=False)
df_p.loc[[4473,7227],'target']
df_p_extract = df_p[df_p['text'].str.contains(t_text,regex=False)]
df_p_extract
p_text_chunk_list = []
p_text_index_list = []

chunk_size = 300

for p_index, p_row in df_p_extract.iterrows():
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
df_p_chunked[df_p_chunked['text'].str.contains(t_text,regex=False)]
test.loc[31525]
t_text = test.loc[31525,'text'].translate(translation_1)
t_text
df_p_extract = df_p.loc[[11001,13188]]
p_text_chunk_list = []
p_text_index_list = []

chunk_size = 300

for p_index, p_row in df_p_extract.iterrows():
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
df_p_chunked[df_p_chunked['p_index'] == 11001]
df_p_chunked[df_p_chunked['p_index'] == 13188]
df_p_chunked[df_p_chunked['p_index'] == 11001].iloc[-1]['text'] == df_p_chunked[df_p_chunked['p_index'] == 13188].iloc[-1]['text']
df_p_chunked[df_p_chunked['p_index'] == 11001].iloc[-1]['text'] == t_text