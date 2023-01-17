# The essentials

import numpy as np

import pandas as pd



# Plotting

%matplotlib inline

import matplotlib.pyplot as plt



# Std lib

from collections import defaultdict, Counter
# load text and ciphertexts in pandas dataframe

train = pd.read_csv('../input/training.csv', index_col='index')

train['length'] = train['text'].apply(lambda x: len(x))

# ceil the length of the plain texts and save locally (for matching plain and cipher texts)

train['length_100'] = (np.ceil(train['length'] / 100) * 100).astype(int)

test = pd.read_csv('../input/test.csv')

test['length'] = test['ciphertext'].apply(lambda x: len(x))
# alphabets and key

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
# encrypt to difficulty 1

train['cipher1'] = train['text'].apply(encrypt_step1)

train.head()
# select difficulty 2 ciphertexts

diff2 = test[test['difficulty'] == 2]

# group the ciphertexts by length & sort the values 

lengths = diff2.groupby('length')['ciphertext'].count().sort_values()

# search for those cipher lengths which only once in our ciphertexts set

rare_lengths =  lengths[lengths == 1].index

# match them with the train (plaintext) set and count how many times we found a plaintext matching the length of the ciphertexts

train[train['length_100'].isin(rare_lengths)].groupby('length_100')['text'].count()
matches = [7100, 7900]

train[train['length_100'].isin(matches)].sort_values('length_100')
diff2[diff2['length'].isin(matches)].sort_values('length')
print("Cipher1 text: ", train[train.plaintext_id=="ID_44394ca71"].cipher1.values[0][0:35], "(generated from the plaintext)")

print("Cipher2 text: ",test[test.ciphertext_id=="ID_f8d497eb8"].ciphertext.values[0][0:35])
print("Cipher1 text: ",train[train.plaintext_id=="ID_44394ca71"].cipher1.values[0][0:35])

print("Cipher2 text: ",test[test.ciphertext_id=="ID_f8d497eb8"].ciphertext.values[0][(55//2):(55//2)+35])
cipher1 = train[train.plaintext_id=="ID_44394ca71"].cipher1.values[0][0:35]

cipher2 = test[test.ciphertext_id=="ID_f8d497eb8"].ciphertext.values[0][(55//2):(55//2)+35]



diff_char1 = ""

diff_char2 = ""

for i in range(len(cipher1)):

    if cipher1[i] != cipher2[i]:

        diff_char1 += cipher1[i]

        diff_char2 += cipher2[i]



print(diff_char1)

print(diff_char2)
def find_key(cipher2, cipher1, alphabet):

    ciphertext = ''

    for i, c in enumerate(cipher2):

        # check if character is in alphabet

        if c in alphabet:

            # get the index of the cipher2 character in the alphabet

            plain_key = alphabet.index(cipher2[i])

            # do the same for the cipher1 character

            enc_key = alphabet.index(cipher1[i])

            # subtract, but make sure we are still inside the alphabet

            newIndex = (plain_key - enc_key) % len(alphabet)

            # return character from alphabet based on subtracted indices

            ciphertext += alphabet[newIndex]

            #cntr = (cntr + 1) % key_length

        else:

            ciphertext += ""

            

    return ciphertext



find_key(diff_char2, diff_char1, "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
find_key(diff_char2, diff_char1, 'aAbBcCdDeEfFgGhHiIjJkKlLmMnNoOpPqQrRsStTuUvVwWxXyYzZ')
def encrypt_vigenere(plaintext, key, alphabet):

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



def decrypt_vigenere(plaintext, key, alphabet):

    key_length = len(key)

    cntr = 0

    ciphertext = ''

    for i, c in enumerate(plaintext):

        if c in alphabet:

            charIndex = alphabet.index(c)

            keyIndex = alphabet.index(key[cntr])

            newIndex = (charIndex - keyIndex) % len(alphabet)

            ciphertext += alphabet[newIndex]

            cntr = (cntr + 1) % key_length

        else:

            ciphertext += c

            

    return ciphertext
cipher = test[test.ciphertext_id=="ID_f8d497eb8"].ciphertext.values[0]



step1 = decrypt_vigenere(cipher, 'xenophon', 'aAbBcCdDeEfFgGhHiIjJkKlLmMnNoOpPqQrRsStTuUvVwWxXyYzZ')

step2 = decrypt_step1(step1)



# decrypted text

print(step2[0:76])

# plaintext

print("                          ",train[train.plaintext_id=="ID_44394ca71"].text.values[0][0:76-27])