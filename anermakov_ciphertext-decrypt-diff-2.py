import pandas as pd

import numpy as np
%load_ext autoreload

%autoreload 2
df_train = pd.read_csv('../input/ciphertext-challenge-iii/train.csv')

df_train.shape
# Encrypt and decrypt of diff=1



ALPHABET = 'abcdefghijklmnopqrstuvwxy'

KEY = 'pyle'





def __encrypt_char(c, alphabet, key, idx):

    key_size, alphabet_size = len(key), len(alphabet)

    if c in alphabet:

        pos = (alphabet.index(c) + alphabet.index(key[idx])) % alphabet_size

    return alphabet[pos], (idx + 1) % key_size





def __decrypt_char(c, alphabet, key, idx):

    key_size, alphabet_size = len(key), len(alphabet)

    if c in alphabet:

        pos = (alphabet_size + alphabet.index(c) - alphabet.index(key[idx])) % alphabet_size

    return alphabet[pos], (idx + 1) % key_size





def encrypt1(text, alphabet=ALPHABET, key=KEY, shift=0, verbose=False):

    text = ''.join(['a']*shift) + text

    key_size, alphabet_size = len(key), len(alphabet)

    alphabet1, alphabet2 = alphabet.lower(), alphabet.upper()

    key1, key2 = key.lower(), key.upper()

    idx = 0

    ciphertext = ''

    for c in text:

        if verbose:

            print(c, end='')

        if c in alphabet1:

            c, idx = __encrypt_char(c, alphabet1, key1, idx)

        elif c in alphabet2:

            c, idx = __encrypt_char(c, alphabet2, key2, idx)

        if verbose:

            print('->', c)

        ciphertext += c

    return ciphertext[shift:]





def decrypt1(ciphertext, alphabet=ALPHABET, key=KEY, verbose=False):

    key_size, alphabet_size = len(key), len(alphabet)

    alphabet1, alphabet2 = alphabet.lower(), alphabet.upper()

    key1, key2 = key.lower(), key.upper()

    idx = 0

    text = ''

    for c in ciphertext:

        if verbose:

            print(c, end='')

        if c in alphabet1:

            c, idx = __decrypt_char(c, alphabet1, key1, idx)

        elif c in alphabet2:

            c, idx = __decrypt_char(c, alphabet2, key2, idx)

        if verbose:

            print('->', c)

        text += c

    return text
for shift in range(4):

    df_train[f'text{shift}'] = (''.join(['a']*shift) + df_train['text']).apply(lambda x: encrypt1(x)[shift:])
df_train.head()
df_test = pd.read_csv('../input/ciphertext-challenge-iii/test.csv')

df_test.shape
df_test.head()
df_train['text_len'] = df_train['text'].apply(lambda x: len(x))

df_test['ciphertext_len'] = df_test['ciphertext'].apply(lambda x: len(x))
mask2 = df_test['difficulty'] == 2

df_test = df_test[mask2]

df_test.head()
df_test['ciphertext_len'].value_counts()
df_test['ciphertext_len'].sort_values().tail(10)
ciphertext1_idx = df_test['ciphertext_len'].sort_values().index[-1]
df_test.loc[ciphertext1_idx, 'ciphertext']
df_train['text_len'].sort_values().tail(10)
df_train.loc[df_train['text_len'].sort_values().tail(10).index, 'text']
text1_idx = df_train['text_len'].sort_values().tail(10).index[-1]

df_train.loc[text1_idx, 'text3']
text1 = df_train.loc[text1_idx, 'text3']

s_text1 = pd.Series(list(text1))
ciphertext1 = df_test.loc[ciphertext1_idx, 'ciphertext']

s_ciphertext1 = pd.Series(list(ciphertext1))
import matplotlib.pyplot as plt
plt.figure(figsize=(12,6))

s_text1.value_counts(normalize=True).plot(kind='bar', log=True, alpha=0.25, label='train')

s_ciphertext1.value_counts(normalize=True).plot(kind='bar', log=True, color='orange', alpha=0.25, label='test')

plt.legend()
def stats(s):

    '''Print text statistics for different characters'''

    print('lower:', len([x for x in s if x.islower()]), end=' ')

    print('upper:', len([x for x in s if x.isupper()]), end=' ')

    print('digit:', len([x for x in s if x.isdigit()]), end=' ')

    print('space:', len([x for x in s if x == ' ']), end=' ')

    print(', -', len([x for x in s if x == ',']), end=' ')

    print('. -', len([x for x in s if x == '.']))

    print('GPLBIXTE:', len([x for x in s if x in 'GPLBIXTE']), end=' ')

    print('GPQOMDYW:', len([x for x in s if x in 'GPQOMDYW']), end=' ')

    print('U:', len([x for x in s if x in 'U']))
stats(text1)
stats(ciphertext1)
pd.Series([x for x in text1 if x.isupper()]).value_counts()
pd.Series([x for x in ciphertext1 if x.isupper()]).value_counts()
len(text1), len(ciphertext1)
dot_p = [pos for pos, char in enumerate(text1) if char == '.']

dot_c = [pos for pos, char in enumerate(ciphertext1) if char == '.']

dot_U = [pos for pos, char in enumerate(ciphertext1) if char == 'U']
plt.title('Dots postitions')

plt.step(range(len(dot_p)), dot_p, label='text')

plt.step(range(len(dot_c)), dot_c, label='ciphertext')

plt.legend()
comma_p = [pos for pos, char in enumerate(text1) if char == ',']

comma_c = [pos for pos, char in enumerate(ciphertext1) if char == ',']
plt.title('Commas postitions')

plt.step(range(len(comma_p)), comma_p, label='text')

plt.step(range(len(comma_c)), comma_c, label='ciphertext')

plt.legend()
U_p = [pos for pos, char in enumerate(text1) if char == 'U']

U_c = [pos for pos, char in enumerate(ciphertext1) if char == 'U']
plt.title('U postitions')

plt.step(range(len(U_p)), U_p, where='post', label='text')

plt.plot(range(len(U_p)), U_p, 'C0o')

plt.step(range(len(U_c)), U_c, where='post', label='ciphertext')

plt.plot(range(len(U_c)), U_c, 'C1o')

plt.legend()
ciphertext2_idx = df_test['ciphertext_len'].sort_values().index[-2]

ciphertext3_idx = df_test['ciphertext_len'].sort_values().index[-3]
ciphertext2 = df_test.loc[ciphertext2_idx, 'ciphertext']

ciphertext2
ciphertext3 = df_test.loc[ciphertext3_idx, 'ciphertext']

ciphertext3
df_train[(df_train['text_len'] > 300) & (df_train['text_len'] <= 400)]
text2_idx = 14716

text2 = df_train.loc[text2_idx, 'text1']

text2
text3_idx = 61720

text3 = df_train.loc[text3_idx, 'text0']

text3
s_text123 = pd.Series([y for x in [text1, text2, text3] for y in list(x)])

s_ciphertext123 = pd.Series([y for x in [ciphertext1, ciphertext2, ciphertext3] for y in list(x)])
plt.figure(figsize=(12,6))

s_text1.value_counts(normalize=True).plot(kind='bar', log=True, alpha=0.25, label='train')

s_ciphertext1.value_counts(normalize=True).plot(kind='bar', log=True, color='orange', alpha=0.25, label='test')

plt.legend()
stats(text2)
stats(ciphertext2)
stats(text3)
stats(ciphertext3)
text3
def size(text, other_side):

    '''Detecting size of rectangle to fit text, using given other side'''

    return (len(text) - 1) // other_side + 1
def reshape(text, n_rows=None, n_cols=50, pattern=None, wide=True, verbose=False):

    '''Reshape text into rectangle by rows or columns with output'''

    if verbose:

        print(text, '\n')

    if n_rows is not None:

        if verbose:

            row_size = size(text, n_rows)

            for i in range(n_rows):

                row = text[i*row_size:(i+1)*row_size]

                if wide:

                    print(' '.join(list(row)))        

                else:

                    print(row)

        return text

    res = ''

    n_rows = size(text, n_cols)

    for i in range(n_rows):

        row = text[i::n_rows]

        res += row

        if verbose:

            if wide:

                print(' '.join(list(row)))

            else:

                print(row)

    if pattern is not None:

        if pattern in res:

            print(pattern, '->->->', n_cols)

        if pattern[::-1] in res:

            print(pattern, '<-<-<-', n_cols)

    return res
# np.array(list(ciphertext3)).reshape(20, -1).T
def find_pattern(cipertext, patterns):

    '''Find given patterns in ciphertext in direct or backward direction for different key size'''

    for key_size in [5, 10, 20, 25, 50, 100]:

        print('----- key_size -----', key_size)

        for pattern in patterns:

            reshape(cipertext, n_cols=key_size, pattern=pattern)
find_pattern(ciphertext3, ['BEVD', 'RWIDCPV', 'JTMFSD', 'BLMKR'])
text3
reshape(ciphertext3, n_cols=20, verbose=True)
reshape(ciphertext3, n_rows=20, verbose=True)
text3
from PIL import Image

img = Image.open('../input/diff2-mapping/photo5219986312940071919.jpg')

img
def shift_row(row_idx, text, n_cols, shift):

    row = text[row_idx*n_cols:(row_idx+1)*n_cols]

    return row[shift:] + row[:shift]
def decrypt(ciphertext, n_cols=20, verbose=False):

    '''First version'''

    #1. extract head and tail rows

    d = n_cols//2

    head, tail = ciphertext[:d], ciphertext[-d:]

    if verbose:

        print('head:', head, 'tail:', tail)

    #2. swap right and left cols

    ciphertext_ = ''

    n_rows = size(ciphertext, n_cols)

    for i in range(n_rows):

        ciphertext_ += shift_row(i, ciphertext, n_cols, d)

    ciphertext = ciphertext_

    #3. extracting cols

    cols = []

    for i in range(n_cols):

        col = ciphertext[i::n_cols]

        if i < d:

            col = col[:-1]  

        else:

            col = col[1:]

        if i % 2 == 0:

            col = head[i//2] + col

        else:

            col = tail[i//2] + col[::-1]

        if verbose:

            print('i:', i, 'col:', col)

        cols += col

    return ''.join(cols)
text3d = decrypt(ciphertext3, verbose=True)
text3
text3d
text3 in text3d
ciphertext3[:10], ciphertext3[-10:]
reshape(ciphertext3[10:-10], n_rows=19, verbose=True)
def decrypt(ciphertext, n_cols=20, verbose=False):

    '''Simplified version'''

    #1. extract head and tail rows

    n_rows = size(ciphertext, n_cols)

    d = n_rows//2 

    s, f = (d, d) if n_rows % 2 == 0 else (d + 1, d)

    head, tail, ciphertext = ciphertext[:s], ciphertext[-f:], ciphertext[s:-f]

    if verbose:

        print('head:', head, 'tail:', tail)

    #2. extracting rows

    chars = []

    for i in range(n_rows):

        row = ciphertext[i::n_rows]

        if i % 2 == 0:

            row = head[i//2] + row

        else:

            row = tail[i//2] + row[::-1]

        if verbose:

            print(f'i: {i:2}', 'row:', row)

        chars += row

    return ''.join(chars)
text3d = decrypt(ciphertext3, verbose=True)
text3 in text3d
def encrypt(text, n_cols=20, verbose=False):

    #1. constructing cols, head and tail

    n_rows = size(text, n_cols)

    head, tail = text[::n_cols*2], text[n_cols::n_cols*2]

    if verbose:

        print('head:', head, 'tail:', tail)

    cols = []

    for i in range(n_rows):

        col = text[i*n_cols:(i+1)*n_cols]

        if i % 2 == 0:

            col = col[1:]

        else:

            col = col[1:][::-1]

        if verbose:

            print(f'i: {i:2}', 'col:', col)

        cols += [col]

    #2. convert cols to rows

    ciphertext = ''

    for i in range(n_cols-1):

        ciphertext += ''.join([col[i] for col in cols])

    #3. add head and tail

    return head + ciphertext + tail
ciphertext3e = encrypt(text3d, 20, verbose=True)
ciphertext3e == ciphertext3
text2
find_pattern(ciphertext2, ['NLXQAR', 'JYXSHY', 'PYDSD', 'XYJHGW'])
text2d = decrypt(ciphertext2, 20, verbose=True)
text2
text2 in text2d
text1
find_pattern(ciphertext1, ['LHTM', 'UKDPR', 'VAPIDK'])
reshape(ciphertext1, n_rows=20, verbose=True)
reshape(ciphertext1[28:-27], n_rows=19, verbose=True)
text1d = decrypt(ciphertext1, verbose=True)
text1
text1 in text1d
ciphertext1e = encrypt(text1d, verbose=True)
ciphertext1e == ciphertext1
decrypt1(decrypt(ciphertext2))
df_test['text'] = df_test['ciphertext'].apply(lambda x: decrypt(x)).apply(lambda x: decrypt1(x))
from tqdm import tqdm_notebook
df_test['train_index'] = 0



for idx, train_row in tqdm_notebook(df_train.iterrows(), total=df_train.shape[0]):

    l = len(train_row['text'])

    text_trimmed = df_test['text'].apply(lambda x: x[(len(x)-l)//2:(len(x)-l)//2+l])

    cond = text_trimmed == train_row['text']

    if df_test[cond].shape[0] == 1:

        print('.', end='')

        df_test.loc[cond, 'train_index'] = train_row['index']
df_test.head()
df_test[df_test['train_index'] == 0]
df_test.rename(columns={'train_index':'index'}, inplace=True)
df_subm = pd.read_csv('../input/ciphertext-challenge-iii/test.csv', usecols=['ciphertext_id'])
df_subm = df_subm[['ciphertext_id']].merge(df_test[['ciphertext_id', 'index']], how='left',

                                           left_on='ciphertext_id', right_on='ciphertext_id')

df_subm = df_subm.fillna(0)

df_subm['index'] = df_subm['index'].astype(np.int32)
df_subm.head(10)
df_subm.to_csv('diff2.csv', index=None)