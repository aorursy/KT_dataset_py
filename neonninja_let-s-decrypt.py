import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from collections import Counter

from tqdm.auto import tqdm

tqdm.pandas()

import string

import re

pd.set_option('display.max_colwidth', -1)
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

# Create a column for the lengths of the texts

train['length'] = train.text.apply(len)

train['n_words'] = train.text.str.split().apply(len)

train = train.sort_values(by='length', ascending=False)

test['length'] = test.ciphertext.apply(len)

test['n_words'] = test.ciphertext.str.split().apply(len)

# Split the ciphers into their difficulty levels

test1 = test[test.difficulty == 1]

test2 = test[test.difficulty == 2]

test3 = test[test.difficulty == 3]

test4 = test[test.difficulty == 4]

display(train.head())

display(test1.head())
%%time

# From https://www.kaggle.com/julianb/time-efficient-pairing

wordlist = {}

for i,t in train.iterrows():

    for w in t.text.split():

        if len(w) > 1:

            if w not in wordlist:

                wordlist[w] = 1

            else:

                wordlist[w] += 1

print("Built wordlist frequencies")

rare_map = {}

for i,t in train.iterrows():

    fs = []

    for w in t.text.split():

        if len(w) > 1:

            fs.append((w, wordlist[w]))

    if len(fs) == 0:

        continue

    fs.sort(key=lambda x:x[1])

    for rare_w,_ in fs[:3]:

        if rare_w not in rare_map:

            rare_map[rare_w] = [t]

        else:

            rare_map[rare_w].append(t)

print("Built hash table")

def find(pt):

    fs = []

    for w in pt.split():

        if w in rare_map.keys():

            fs.append((w, wordlist[w]))

    if len(fs)==0:

        return None

    fs.sort(key=lambda x:x[1])

    for rare_w, _ in fs[:5]: #We'll check up to 5 rare words, just to be safe.

        for t in rare_map[rare_w]:

            if t.text in pt:

                return t
sort1 = test1.length.sort_values(ascending=False).head()

longest = test1.ciphertext[sort1.index[0]]

matching_pieces = train[(train.length>=401) & (train.length<=500)]

match = matching_pieces.text.values[0]

pad = int((len(longest) - len(match)) / 2)

longest = longest[pad:-pad]

longest = "".join(c for c in longest if ord(c) >= ord("A"))

match = "".join(c for c in match if ord(c) >= ord("A"))

print(longest + "\n" + match)

# Print the difference for each A-Z, a-z character, wrapping around the alphabet, leaving out z

diff = [(ord(match[i]) - ord(longest[i]) + 25) % 25 for i in range(len(longest))]

for i in range(0, 4*4, 4):

    print(longest[i:i+4] + "->" + match[i:i+4])

    print(diff[i:i+4])
%%time

key = [14, 21, 10, 1]



def decrypt(enc):

    decrypted = []

    i = 2

    for c in enc:

        if c == "z" or c == "Z":

            decrypted.append(c)

        elif c in string.ascii_uppercase:

            ci = string.ascii_uppercase.index(c)

            decrypted.append(string.ascii_uppercase[(ci + key[i % 4]) % 25])

            i += 1

        elif c in string.ascii_lowercase:

            ci = string.ascii_lowercase.index(c)

            decrypted.append(string.ascii_lowercase[(ci + key[i % 4]) % 25])

            i += 1

        else:

            decrypted.append(c)

    return "".join(decrypted)



test1["decrypted"] = test1.ciphertext.progress_apply(decrypt)

test1.head()
%%time

def find_plain_match(row):

    lookup = find(row.decrypted)

    if lookup is None: # no match using rare word lookup - do a search of the possibilities

        possible = train[train.n_words <= row.n_words] # the plaintext version will have equal to or less words (padding might have spaces). We also want the longest match.

        for i,t in possible.iterrows():

            if t.text in row.decrypted:

                lookup = t

                break

    #print("Matched {} with {}".format(row.decrypted, lookup.text))

    return lookup["index"]



test1["index"] = test1.progress_apply(find_plain_match, axis=1)
for d in test1[test1["index"].duplicated(keep=False)].decrypted:

    print(d, find(d).text)
test = test.merge(test1, how="left")

sub = test[["ciphertext_id", "index"]]

sub["index"] = sub["index"].fillna(0).astype(int)

nonzero = sum(sub["index"] != 0)

print("Predicting {}/{} rows - {}%".format(nonzero, len(test), round(nonzero / len(test) * 100, 3)))

sub.to_csv("submission.csv", index=False)

!head submission.csv