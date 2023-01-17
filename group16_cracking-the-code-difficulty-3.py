# The essentials

import numpy as np

import pandas as pd



# Plotting

%matplotlib inline

import matplotlib.pyplot as plt



# Std lib

from collections import defaultdict, Counter



from tqdm import tqdm_notebook
# load text and ciphertexts in pandas dataframe

train = pd.read_csv('../input/training.csv', index_col='index')

train['length'] = train['text'].apply(lambda x: len(x))

# ceil the length of the plain texts and save locally (for matching plain and cipher texts)

train['length_100'] = (np.ceil(train['length'] / 100) * 100).astype(int)

test = pd.read_csv('../input/test.csv')

test['length'] = test['ciphertext'].apply(lambda x: len(x))
# Difficulty 1

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



# Difficulty 2

def encrypt_step2(plaintext, key='xenophon', alphabet='aAbBcCdDeEfFgGhHiIjJkKlLmMnNoOpPqQrRsStTuUvVwWxXyYzZ'):

    key_length = len(key)

    cntr = 0

    ciphertext = ''

    for i, c in enumerate(plaintext):

        if c in alphabet:

            charIndex = alphabet.index(c)

            keyIndex = alphabet.index(key[cntr])

            newIndex = (charIndex + keyIndex) % len(alphabet)

            ciphertext += alphabet[newIndex]

            cntr = (cntr + 1) % key_length

        else:

            ciphertext += c

            

    return ciphertext



def decrypt_step2(ciphertext, key='xenophon', alphabet='aAbBcCdDeEfFgGhHiIjJkKlLmMnNoOpPqQrRsStTuUvVwWxXyYzZ'):

    key_length = len(key)

    cntr = 0

    plaintext = ''

    for i, c in enumerate(ciphertext):

        if c in alphabet:

            charIndex = alphabet.index(c)

            keyIndex = alphabet.index(key[cntr])

            newIndex = (charIndex - keyIndex) % len(alphabet)

            plaintext += alphabet[newIndex]

            cntr = (cntr + 1) % key_length

        else:

            plaintext += c

            

    return plaintext

# Count occurences of characters of ciphertexts of difficulty 2

diff2_cntr = Counter(''.join(test[test['difficulty'] == 2]['ciphertext'].values))

# new dataframe with frequency and letter

diff2_stats = pd.DataFrame([[x[0], x[1]] for x in diff2_cntr.items()], columns=['Letter', 'Frequency'])

# sort dataframe on occurence of frequency

diff2_stats = diff2_stats.sort_values(by='Frequency', ascending=True)



# plot

f, ax = plt.subplots(figsize=(5, 15))

plt.barh(np.array(range(len(diff2_stats))) + 0.5, diff2_stats['Frequency'].values)

plt.yticks(np.array(range(len(diff2_stats))) + 0.5, diff2_stats['Letter'].values)

plt.show()
# Count occurences of characters of ciphertexts of difficulty 3

diff3_cntr = Counter(''.join(test[test['difficulty'] == 3]['ciphertext'].values))

# new dataframe with frequency and letter

diff3_stats = pd.DataFrame([[x[0], x[1]] for x in diff3_cntr.items()], columns=['Letter', 'Frequency'])

# sort dataframe on occurence of frequency

diff3_stats = diff3_stats.sort_values(by='Frequency', ascending=True)



# plot

f, ax = plt.subplots(figsize=(5, 15))

plt.barh(np.array(range(len(diff3_stats))) + 0.5, diff3_stats['Frequency'].values)

plt.yticks(np.array(range(len(diff3_stats))) + 0.5, diff3_stats['Letter'].values)

plt.show()
# Let's get all plaintexts of length 700 (this can be done with any length % 100 == 0)

train_filtered = train[train['length'] == 700]

train_filtered['cipher2'] = train_filtered['text'].apply(lambda x: encrypt_step2(encrypt_step1(x)))



plain_to_cipher = defaultdict(list)

for i, row in tqdm_notebook(train_filtered.iterrows(), total=len(train_filtered)):

    row_cntr = Counter(row['cipher2'])

    test_filtered = test[(test['length'] == len(row['text'])) & (test['difficulty'] == 3)]

    for j, row2 in test_filtered.iterrows():

        row2_cntr = Counter(row2['ciphertext'])

        

        match = True

        for k in row_cntr:

            if k not in row2_cntr or row_cntr[k] != row2_cntr[k]:

                match = False

                break

                

        if match:

            plain_to_cipher[i].append(j)

            break

            

print(plain_to_cipher)
mapping = defaultdict(set)

for k in tqdm_notebook(plain_to_cipher):

    plain_encrypted = train_filtered.loc[k, 'cipher2']

    plain_cntr = Counter(plain_encrypted)

    

    cipher = test.loc[plain_to_cipher[k][0], 'ciphertext']

    cipher_cntr = Counter(cipher)

    

    for key in plain_cntr:

        if plain_cntr[key] == 1 and cipher_cntr[key] == 1:

            mapping[cipher.index(key)].add(plain_encrypted.index(key))

            

for key in sorted(mapping.keys()):

    print(key, mapping[key])
def create_table(length, n_cols=22):

    n_rows = int(np.ceil(length / n_cols))

    table = np.zeros((n_rows, n_cols))

    table[:, 0] = list(range(n_rows))

    

    cntr = n_rows - 1

    for col in range(1, n_cols // 2):

        for row in range(n_rows):

            if 0 < (length % n_cols) <= col and row == n_rows - 1:

                table[row, col] = np.NaN

            else:

                cntr += 1

                table[row, col] = cntr

                

            

            if 0 < (length % n_cols) <= 22 - col and row == n_rows - 1:

                table[row, 22 - col] = np.NaN

            else:

                cntr += 1

                table[row, 22 - col] = cntr

            

    for row in range(n_rows):

        if 0 < (length % n_cols) <= n_cols // 2 and row == n_rows - 1:

            table[row, n_cols // 2] = np.NaN

        else:

            cntr += 1

            table[row, n_cols // 2] = cntr

        

    return table



np.set_printoptions(edgeitems=30, linewidth=100000, 

                    formatter=dict(float=lambda x: "% 4g" % x))

create_table(300)



# We now number the cells in this table from 0 -> L, starting from the left and top and going from left to right and from top to bottom.

# For length 300, the mapping thus becomes:

# * Map character k in 0 - 13 on 22 * k

# * Map character 14 on 1

# * Map character 15 on 21

# * Map character 16 on 23

# * Map character 17 on 43

# ...
def create_map(text, n_cols=22):

    table = create_table(len(text), n_cols=n_cols)

    mapping = {}

    rev_mapping = {}

    for i, c in enumerate(text):

        row_ix, col_ix = [x[0] for x in np.where(table == i)]

        new_pos = row_ix * n_cols + col_ix

        mapping[i] = new_pos

        rev_mapping[new_pos] = i



    return mapping, rev_mapping



def encrypt_step3(text):

    mapping, _ = create_map(text)

    output = ''

    for i, c in enumerate(text):

        output += text[mapping[i]]

    return output



def decrypt_step3(text):

    _, rev_mapping = create_map(text)

    output = ''

    for i, c in enumerate(text):

        output += text[rev_mapping[i]]

    return output
cipher = test[(test['difficulty'] == 3)].sample(1).iloc[0, :]['ciphertext']

step1 = decrypt_step3(cipher)

step2 = decrypt_step2(step1)

step3 = decrypt_step1(step2)

print(step3)