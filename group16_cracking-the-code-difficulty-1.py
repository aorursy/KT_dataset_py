# Basic import statements

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from collections import Counter



%matplotlib inline

import matplotlib.pyplot as plt
# load text and ciphertexts in pandas dataframe

train = pd.read_csv('../input/training.csv', index_col='index')

train['length'] = train['text'].apply(lambda x: len(x))

# ceil the length of the plain texts and save locally (for matching plain and cipher texts)

train['length_100'] = (np.ceil(train['length'] / 100) * 100).astype(int)

test = pd.read_csv('../input/test.csv')

test['length'] = test['ciphertext'].apply(lambda x: len(x))
# select difficulty 1 ciphertexts

diff1 = test[test['difficulty'] == 1]

# group the ciphertexts by length & sort the values 

lengths = diff1.groupby('length')['ciphertext'].count().sort_values()

# search for those cipher lengths which only once in our ciphertexts set

rare_lengths =  lengths[lengths == 1].index

# match them with the train (plaintext) set and count how many times we found a plaintext matching the length of the ciphertexts

train[train['length_100'].isin(rare_lengths)].groupby('length_100')['text'].count()
matches = [7300, 7700, 8500, 14200]

train[train['length_100'].isin(matches)].sort_values('length_100')
diff1[diff1['length'].isin(matches)].sort_values('length')
# Count occurences of charcters in the train plaintext (we used the ID_4929f84c6 plain text from the previous analysis)

plain_char_cntr = Counter(''.join(train[train.plaintext_id=='ID_4929f84c6']['text'].values))

# new dataframe with frequency and letter

plain_stats = pd.DataFrame([[x[0], x[1]] for x in plain_char_cntr.items()], columns=['Letter', 'Frequency'])

# sort dataframe on occurence of frequency

plain_stats = plain_stats.sort_values(by='Frequency', ascending=True)



# plot

f, ax = plt.subplots(figsize=(5, 15))

plt.barh(np.array(range(len(plain_stats))) + 0.5, plain_stats['Frequency'].values)

plt.yticks(np.array(range(len(plain_stats))) + 0.5, plain_stats['Letter'].values)

plt.show()



# Space is the most occurring character, folowed by e, t, a, ...
# same approach for ciphertext with id 'ID_a6ecf7480'

cipher_char_cntr = Counter(''.join(test[test['ciphertext_id'] == 'ID_a6ecf7480']['ciphertext'].values))

cipher_stats = pd.DataFrame([[x[0], x[1]] for x in cipher_char_cntr.items()], columns=['Letter', 'Frequency'])

cipher_stats = cipher_stats.sort_values(by='Frequency', ascending=True)



f, ax = plt.subplots(figsize=(5, 15))

plt.barh(np.array(range(len(cipher_stats))) + 0.5, cipher_stats['Frequency'].values)

plt.yticks(np.array(range(len(cipher_stats))) + 0.5, cipher_stats['Letter'].values)

plt.show()



# The bars match the training distribution very well! Most occurring character here is 7, followed by l, x, 4, ...
# merge plaintext frequency stats together with the cipher text stats based on the Frequency scores

freq_alphabet = pd.merge(plain_stats, cipher_stats, on=['Frequency'])

# sort dataframe on frequency score

freq_alphabet = freq_alphabet.sort_values(by='Frequency', ascending=False)

# print first 20 rows of this dataframe

freq_alphabet.head(20)
# Manually fix the mapping for the remainining characters

alphabet = """7lx4v!2oQ[O=,yCzV:}dFX#(Wak/bqne*JApK{cmf6 GZDj9gT\'"YSHiE]5)81hMNwI@P?Us%;30uBrLR-.$t"""

key =      """ etaoinsrhldcumfygwpb.v,kI\'T"A-SBMxDHj)CW(ELORN!FGPJz0qK?1VY:U92/3*5;478QZ6X%$}#@={[]"""



decrypt_mapping = {}

encrypt_mapping = {}

for i, j in zip(alphabet, key):

    decrypt_mapping[ord(i)] = ord(j)

    encrypt_mapping[ord(j)] = ord(i)



def encrypt_step1(x):

    return x.translate(encrypt_mapping)



def decrypt_step1(x):

    return x.translate(decrypt_mapping)
cipher = test[(test['difficulty'] == 1)].sample(1).iloc[0, :]['ciphertext']

print(decrypt_step1(cipher))