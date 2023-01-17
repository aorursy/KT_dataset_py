import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

from collections import Counter



# load the data

train_df = pd.read_csv('../input/ciphertext-challenge-iii/train.csv', index_col='plaintext_id')

test_df = pd.read_csv('../input/ciphertext-challenge-iii/test.csv', index_col='ciphertext_id')

submission = pd.read_csv('../input/ciphertext-challenge-iii/sample_submission.csv', index_col='ciphertext_id')



#  create a 'length' column

train_df['length'] = train_df.text.apply(len)

test_df['length'] = test_df.ciphertext.apply(len)



# cypher level

test_df_level_1 = test_df[test_df.difficulty==1].copy()

test_df_level_2 = test_df[test_df.difficulty==2].copy()

test_df_level_3 = test_df[test_df.difficulty==3].copy()

test_df_level_4 = test_df[test_df.difficulty==4].copy()
# plain text

plain_char_cntr = Counter(''.join(train_df['text'].values))

plain_stats = pd.DataFrame([[x[0], x[1]] for x in plain_char_cntr.items()], columns=['Letter', 'Frequency'])

plain_stats = plain_stats.sort_values(by='Frequency', ascending=False)



f, ax = plt.subplots(figsize=(15, 5))

ax.title.set_text('plaintext')

plt.bar(np.array(range(len(plain_stats))) + 0.5, plain_stats['Frequency'].values)

plt.xticks(np.array(range(len(plain_stats))) + 0.5, plain_stats['Letter'].values)

plt.show()



# cipher text - level 1

cipher_char_cntr = Counter(''.join(test_df_level_1['ciphertext'].values))

cipher_stats = pd.DataFrame([[x[0], x[1]] for x in cipher_char_cntr.items()], columns=['Letter', 'Frequency'])

cipher_stats = cipher_stats.sort_values(by='Frequency', ascending=False)



f, ax = plt.subplots(figsize=(15, 5))

ax.title.set_text('ciphertext level 1')

plt.bar(np.array(range(len(cipher_stats))) + 0.5, cipher_stats['Frequency'].values)

plt.xticks(np.array(range(len(cipher_stats))) + 0.5, cipher_stats['Letter'].values)

plt.show()



# cipher text - level 2

cipher_char_cntr = Counter(''.join(test_df_level_2['ciphertext'].values))

cipher_stats = pd.DataFrame([[x[0], x[1]] for x in cipher_char_cntr.items()], columns=['Letter', 'Frequency'])

cipher_stats = cipher_stats.sort_values(by='Frequency', ascending=False)



f, ax = plt.subplots(figsize=(15, 5))

ax.title.set_text('ciphertext level 2')

plt.bar(np.array(range(len(cipher_stats))) + 0.5, cipher_stats['Frequency'].values)

plt.xticks(np.array(range(len(cipher_stats))) + 0.5, cipher_stats['Letter'].values)

plt.show()



# cipher text - level 3

cipher_char_cntr = Counter(''.join(test_df_level_3['ciphertext'].values))

cipher_stats = pd.DataFrame([[x[0], x[1]] for x in cipher_char_cntr.items()], columns=['Letter', 'Frequency'])

cipher_stats = cipher_stats.sort_values(by='Frequency', ascending=False)



f, ax = plt.subplots(figsize=(15, 5))

ax.title.set_text('ciphertext level 3')

plt.bar(np.array(range(len(cipher_stats))) + 0.5, cipher_stats['Frequency'].values)

plt.xticks(np.array(range(len(cipher_stats))) + 0.5, cipher_stats['Letter'].values)

plt.show()



# cipher text - level 4

cipher_char_cntr = Counter(''.join(test_df_level_4['ciphertext'].values))

cipher_stats = pd.DataFrame([[x[0], x[1]] for x in cipher_char_cntr.items()], columns=['Letter', 'Frequency'])

cipher_stats = cipher_stats.sort_values(by='Frequency', ascending=False)



f, ax = plt.subplots(figsize=(15, 5))

ax.title.set_text('ciphertext level 4')

plt.bar(np.array(range(len(cipher_stats))) + 0.5, cipher_stats['Frequency'].values)

plt.xticks(np.array(range(len(cipher_stats))) + 0.5, cipher_stats['Letter'].values)

plt.show()
# use the length of some text pieces to understand the first cypher level

test_df_level_1.length.sort_values(ascending=False).head()



# then we look in the training data to find the passage with the corresponding length

matching_pieces_1 = train_df[(train_df.length>=401) & (train_df.length<=500)]

matching_pieces_2 = train_df[(train_df.length>=301) & (train_df.length<=400)]



print('Plain text:\n', train_df.loc['ID_f000cad17'].text, '\n\nCiphered text (level 1):\n', 

      test_df_level_1.loc['ID_6100247c5'].ciphertext)



print('\n---------------------------------------------------------\n')

# Let's do the same thing for the second piece of text now

print('Plain text:\n', train_df.loc['ID_ad64b5b8d'].text, '\n\nCiphered text (level 1):\n', test_df_level_1.loc['ID_31bd699f6'].ciphertext)
# Functions to decrypt and encrypt from/to level 1

KEYLEN = len('pyle')

def decrypt_level_1(ctext):

    key = [ord(c) - ord('a') for c in 'pyle']

    key_index = 0

    plain = ''

    for c in ctext:

        cpos = 'abcdefghijklmnopqrstuvwxy'.find(c)

        if cpos != -1:

            p = (cpos - key[key_index]) % 25

            pc = 'abcdefghijklmnopqrstuvwxy'[p]

            key_index = (key_index + 1) % KEYLEN

        else:

            cpos = 'ABCDEFGHIJKLMNOPQRSTUVWXY'.find(c)

            if cpos != -1:

                p = (cpos - key[key_index]) % 25

                pc = 'ABCDEFGHIJKLMNOPQRSTUVWXY'[p]

                key_index = (key_index + 1) % KEYLEN

            else:

                pc = c

        plain += pc

    return plain



def encrypt_level_1(ptext, key_index=0):

    key = [ord(c) - ord('a') for c in 'pyle']

    ctext = ''

    for c in ptext:

        pos = 'abcdefghijklmnopqrstuvwxy'.find(c)

        if pos != -1:

            p = (pos + key[key_index]) % 25

            cc = 'abcdefghijklmnopqrstuvwxy'[p]

            key_index = (key_index + 1) % KEYLEN

        else:

            pos = 'ABCDEFGHIJKLMNOPQRSTUVWXY'.find(c)

            if pos != -1:

                p = (pos + key[key_index]) % 25

                cc = 'ABCDEFGHIJKLMNOPQRSTUVWXY'[p]

                key_index = (key_index + 1) % KEYLEN

            else:

                cc = c

        ctext += cc

    return ctext



def test_decrypt_level_1():

    c_id = 'ID_4a6fc1ea9'

    ciphertext = test_df.loc[c_id]['ciphertext']

    print('Ciphertxt:', ciphertext)

    decrypted = decrypt_level_1(ciphertext)

    print('Decrypted:', decrypted)

    encrypted = encrypt_level_1(decrypted)

    print('Encrypted:', encrypted)

    print("Encrypted == Ciphertext:", encrypted == ciphertext)



test_decrypt_level_1()    
import tqdm

# Make a dictionary for fast lookup of plaintext

plain_dict = {}

for p_id, row in train_df.iterrows():

    text = row['text']

    plain_dict[text] = p_id

print(len(plain_dict))



# Update submission with level 1 decrypted matching texts

matched, unmatched = 0, 0

for c_id, row in tqdm.tqdm(test_df_level_1.iterrows()):

    decrypted = decrypt_level_1(row['ciphertext'])

    found = False

    for pad in range(100):

        start = pad // 2

        end = len(decrypted) - (pad + 1) // 2

        plain_pie = decrypted[start:end]

        if plain_pie in plain_dict:

            p_id = plain_dict[plain_pie]

            row = train_df.loc[p_id]

            submission.loc[c_id] = train_df.loc[p_id]['index']

            matched += 1

            found = True

            break

    if not found:

        unmatched += 1

        print(decrypted)

            

print(f"Matched {matched}   Unmatched {unmatched}")

submission.to_csv('submit-level-1.csv')
# Update sub with level 2 decrypted matching texts

import math

from itertools import cycle



def rail_pattern(n):

    r = list(range(n))

    return cycle(r + r[-2:0:-1])



def encrypt_level_2(plaintext, rails=21):

    p = rail_pattern(rails)

    # this relies on key being called in order, guaranteed?

    return ''.join(sorted(plaintext, key=lambda i: next(p)))

def decrypt_level_2(ciphertext, rails=21):

    p = rail_pattern(rails)

    indexes = sorted(range(len(ciphertext)), key=lambda i: next(p))

    result = [''] * len(ciphertext)

    for i, c in zip(indexes, ciphertext):

        result[i] = c

    return ''.join(result)



matched, unmatched = 0, 0

for c_id, row in tqdm.tqdm(test_df_level_2.iterrows()):

    decrypted = decrypt_level_1(decrypt_level_2(row['ciphertext']))

    found = False

    for pad in range(100):

        start = pad // 2

        end = len(decrypted) - (pad + 1) // 2

        plain_pie = decrypted[start:end]

        if plain_pie in plain_dict:

            p_id = plain_dict[plain_pie]

            row = train_df.loc[p_id]

            submission.loc[c_id] = train_df.loc[p_id]['index']

            matched += 1

            found = True

            break

    if not found:

        unmatched += 1

        print(decrypted)

            

print(f"Matched {matched}   Unmatched {unmatched}")

submission.to_csv('submit-level-2.csv')
level_12_train_index = list(submission[submission["index"] > 0]["index"])

print(len(level_12_train_index))

train_df_level_34 = train_df[~train_df["index"].isin(level_12_train_index)].copy()

train_df_level_34.sort_values("length", ascending=False).head(5)
test_df_level_3["nb"] = test_df_level_3["ciphertext"].apply(lambda x: len(x.split(" ")))

test_df_level_3.sort_values("length", ascending=False).head(5)
# found a match for level 3

c_id = 'ID_f0989e1c5' # length = 700

index = 34509 # length = 671

submission.loc[c_id] = index # train_df.loc[p_id]['index']
# Level 4 looks like base64 encoding

import base64



def encode_base64(x):

    return base64.b64encode(x.encode('ascii')).decode()



def decode_base64(x):

    return base64.b64decode(x)



train_df_level_34["nb"] = train_df_level_34["length"].apply(lambda x: math.ceil(x/100)*100)

ratio = test_df_level_3["length"].mean() / train_df_level_34["nb"].mean()

print(ratio)



def get_length(x):

    n = len(decode_base64(x))/ratio

    n = round(n / 100) * 100

    return n



test_df_level_4["nb"] = test_df_level_4["ciphertext"].apply(lambda x: get_length(x)) 

test_df_level_4.sort_values("nb", ascending=False).head(5)
# found a match for level 4

c_id = 'ID_0414884b0' # length = 900

index = 42677 # length = 842

submission.loc[c_id] = index # train_df.loc[p_id]['index']

submission.to_csv('submit-level-34.csv')