# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tqdm



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv', index_col='plaintext_id')

test = pd.read_csv('../input/test.csv', index_col='ciphertext_id')

sub = pd.read_csv('../input/sample_submission.csv', index_col='ciphertext_id')
train['length'] = train.text.apply(len)

test['length'] = test.ciphertext.apply(len)
train[train['length']<=100]['length'].hist(bins=99)
train.head()
test.head(10)
KEYLEN = 4 # len('pyle')

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

    ciphertext = test.loc[c_id]['ciphertext']

    print('Ciphertxt:', ciphertext)

    decrypted = decrypt_level_1(ciphertext)

    print('Decrypted:', decrypted)

    encrypted = encrypt_level_1(decrypted)

    print('Encrypted:', encrypted)

    print("Encrypted == Ciphertext:", encrypted == ciphertext)



test_decrypt_level_1()    
plain_dict = {}

for p_id, row in train.iterrows():

    text = row['text']

    plain_dict[text] = p_id

print(len(plain_dict))
matched, unmatched = 0, 0

for c_id, row in tqdm.tqdm(test[test['difficulty']==1].iterrows()):

    decrypted = decrypt_level_1(row['ciphertext'])

    found = False

    for pad in range(100):

        start = pad // 2

        end = len(decrypted) - (pad + 1) // 2

        plain_pie = decrypted[start:end]

        if plain_pie in plain_dict:

            p_id = plain_dict[plain_pie]

            row = train.loc[p_id]

            sub.loc[c_id] = train.loc[p_id]['index']

            matched += 1

            found = True

            break

    if not found:

        unmatched += 1

        print(decrypted)

            

print(f"Matched {matched}   Unmatched {unmatched}")
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

for c_id, row in tqdm.tqdm(test[test['difficulty']==2].iterrows()):

    decrypted = decrypt_level_1(decrypt_level_2(row['ciphertext']))

    found = False

    for pad in range(100):

        start = pad // 2

        end = len(decrypted) - (pad + 1) // 2

        plain_pie = decrypted[start:end]

        if plain_pie in plain_dict:

            p_id = plain_dict[plain_pie]

            row = train.loc[p_id]

            sub.loc[c_id] = train.loc[p_id]['index']

            matched += 1

            found = True

            break

    if not found:

        unmatched += 1

        print(decrypted)

            

print(f"Matched {matched}   Unmatched {unmatched}")

sub.to_csv('submit-level-2.csv')
level12_train_index = list(sub[sub["index"] > 0]["index"])

print(len(level12_train_index))

train34 = train[~train["index"].isin(level12_train_index)].copy()

test3 = test[test['difficulty']==3].copy()

test4 = test[test['difficulty']==4].copy()

print(train34.shape, test3.shape[0] + test4.shape[0])
test3.sort_values("length", ascending=False).head(5)
test3["nb"] = test3["ciphertext"].apply(lambda x: len(x.split(" ")))

test3.sort_values("length", ascending=False).head(5)
train34.sort_values("length", ascending=False).head(5)
c_id = 'ID_f0989e1c5' # length = 700

index = 34509 # length = 671

sub.loc[c_id] = index # train.loc[p_id]['index']
test4.sort_values("length", ascending=False).head(5)
test4.head(1)["ciphertext"].values[0]
import base64



def encode_base64(x):

    return base64.b64encode(x.encode('ascii')).decode()



def decode_base64(x):

    return base64.b64decode(x)



train34["nb"] = train34["length"].apply(lambda x: math.ceil(x/100)*100)

ratio = test3["length"].mean() / train34["nb"].mean()

print(ratio)



def get_length_level1(x):

    n = len(decode_base64(x))/ratio

    n = round(n / 100) * 100

    return n



train34.head(3)
test4["nb"] = test4["ciphertext"].apply(lambda x: get_length_level1(x)) 

test4.sort_values("nb", ascending=False).head(5)
c_id = 'ID_0414884b0' # length = 900

index = 42677 # length = 842

sub.loc[c_id] = index # train.loc[p_id]['index']
sub.head(3)
def is_correct_mapping(ct_l2, ct_l3):

    tmp = pd.DataFrame([(c,n) for c,n in zip(list(ct_l2), ct_l3.split(" ")) if c.isalpha()])

    tmp.drop_duplicates(inplace=True)

    tmp.columns = ["ch", "num"]

    tmp = tmp.groupby("num")["ch"].nunique()

    return tmp.shape[0] == tmp.sum()



def pad_str(s, special_char = '?'):

    nb = len(s)

    nb_round = math.ceil(nb / 100) * 100

    nb_left = (nb_round - nb) // 2

    nb_right = nb_round - nb - nb_left

    

    left_s = ''.join([special_char] * nb_left)

    right_s = ''.join([special_char] * nb_right)

    return left_s + s + right_s



def is_correct_mapping_low(pt, ct):

    all_ct_l2 = [encrypt_level_2(encrypt_level_1(pad_str(pt), key_index)) for key_index in range(4)]



    for i, ct_l2 in enumerate(all_ct_l2):

        if is_correct_mapping(ct_l2, ct):

            return i

    return -1



def find_mapping(ciphertext_id, ct, train_df):

    nb = len(ct.split(" "))

    nb_low = ((nb // 100) - 1) * 100

    

    rs = []

    selected_rows = train_df[(train_df["length"] > nb_low) & (train_df["length"] < nb)]

    for row_id, row in selected_rows.iterrows():

        pt = row["text"]

        key_index = is_correct_mapping_low(pt, ct)

        if key_index >= 0:

            t = row["index"], key_index

            rs.append(t)

    if len(rs) == 1:

        return rs[0]

    return -1, -1
for ciphertext_id, row in test3[test3["nb"] >= 200].iterrows():

    ct = row["ciphertext"]

    index, key_index = find_mapping(ciphertext_id, ct, train34)

    if index > 0:

        print(ciphertext_id, index, key_index, "(length: {})".format(row["nb"]))

        sub.loc[ciphertext_id] = index # train.loc[p_id]['index']
print(sub[sub["index"] > 0].shape[0], sub[sub["index"] > 0].shape[0]/sub.shape[0])

sub.to_csv('submit-level-2-plus.csv')

sub.head(3)
dict_level3 = {}

for ciphertext_id, row in test3[test3["nb"] >= 200].iterrows():

    ct = row["ciphertext"]

    index, key_index = find_mapping(ciphertext_id, ct, train34)

    if index > 0:

        print(ciphertext_id, index, key_index, "(length: {})".format(row["nb"]))

        dict_level3[ciphertext_id] = (index, key_index) # train.loc[p_id]['index']
dict_level3["ID_11070f053"] = (40234, 1)

dict_level3["ID_c1694eb06"] = (43773, 3)



for ciphertext_id, (index, key_index) in dict_level3.items():

    sub.loc[ciphertext_id] = index

    

print(sub[sub["index"] > 0].shape[0], sub[sub["index"] > 0].shape[0]/sub.shape[0])

sub.to_csv('submit-level-2-plus2.csv')

sub.head(3)
df_mapping = []

special_chars = "?"



def get_mapping(ct_l2, ct):

    tmp = pd.DataFrame([(c,n) for c,n in zip(list(ct_l2), ct.split(" ")) if c not in special_chars])

    tmp.drop_duplicates(inplace=True)

    tmp.columns = ["ch", "num"]

    return tmp



for ciphertext_id, (index, key_index) in dict_level3.items():

    ct = test3.loc[ciphertext_id]["ciphertext"]

    pt = train34[train34["index"]==index]["text"].values[0]

    ct_l2 = encrypt_level_2(encrypt_level_1(pad_str(pt), key_index))

    print(len(ct.split(" ")), len(pt))

    tmp = get_mapping(ct_l2, ct)

    df_mapping.append(tmp)



df_mapping = pd.concat(df_mapping)

print(df_mapping.shape)

df_mapping.head(3)

df_mapping.reset_index(drop=True, inplace=True)

df_mapping.tail(3)
pd.set_option('display.max_rows', 5000)

pd.set_option('display.max_columns', 5000)

pd.set_option('display.max_colwidth', 5000)

pd.set_option('display.width', 5000)



df_ch_num = df_mapping[["ch", "num"]].drop_duplicates().groupby("ch")["num"].apply(list)

df_ch_num = df_ch_num.to_frame("num").reset_index()

df_ch_num["num"] = df_ch_num["num"].apply(lambda x: np.sort([int(n) for n in x]))

df_ch_num["num_alpha"] = df_ch_num["num"].apply(lambda x: np.sort([str(n) for n in x]))

df_ch_num["num_hex"] = df_ch_num["num"].apply(lambda x: np.sort([hex(n) for n in x]))

df_ch_num
from collections import Counter

import matplotlib.pyplot as plt



plt.rcParams["figure.figsize"] = (20,10)
test2 = test[test["difficulty"] == 2].copy()

fullcipher2 = "".join((test2["ciphertext"].values))

dict_fullcipher2 = Counter(fullcipher2)

df_fullcipher2 = pd.DataFrame.from_dict(dict_fullcipher2, orient='index')

df_fullcipher2 = df_fullcipher2.reset_index()

df_fullcipher2.columns = ["ch", "nb"]

df_fullcipher2.sort_values("nb", ascending=False, inplace=True)

print(df_fullcipher2.shape)

df_fullcipher2.head()
print(df_fullcipher2["nb"].mean(), df_fullcipher2["nb"].median())

df_fullcipher2.plot(x="ch", y=["nb"], kind="bar");
fullcipher3 = " ".join((test3["ciphertext"].values))

dict_fullcipher3 = Counter(fullcipher3.split(" "))

df_fullcipher3 = pd.DataFrame.from_dict(dict_fullcipher3, orient='index')

df_fullcipher3 = df_fullcipher3.reset_index()

df_fullcipher3.columns = ["num", "nb"]

df_fullcipher3.sort_values("nb", ascending=False, inplace=True)

print(df_fullcipher3.shape)

df_fullcipher3.head()
df_fullcipher3[df_fullcipher3["nb"] > 1500].plot(x="num", y=["nb"], kind="bar");