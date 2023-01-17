import pandas as pd

import re

from math import ceil
pt = "9]8pVnN4n,DaA6[XNib4K2yVIn[jk[MW0VTo5?J62P?'.0HbpEnter, from one side, LUCIUS, IACHIMO, and  the Roman Army: from the other side, the  British Army, POSTHUMUS LEONATUS following,  like a poor soldier. They march over and go  out. Then enter again, in skirmish, IACHIMO  and POSTHUMUS LEONATUS he vanquisheth and disarmeth IACHIMO, and then leaves himvqlhyWqM4ilXEv]dElTRiO2XBC!)9rl(Iy($HLn'd]ktE6b58y"

print(pt)
plain = "Enter, from one side, LUCIUS, IACHIMO, and  the Roman Army: from the other side, the  British Army, POSTHUMUS LEONATUS following,  like a poor soldier. They march over and go  out. Then enter again, in skirmish, IACHIMO  and POSTHUMUS LEONATUS he vanquisheth and disarmeth IACHIMO, and then leaves him"

print(plain)
plain in pt
train = pd.read_csv('../input/train.csv')

train.text.apply(lambda x: 100*ceil(len(x)/100)).value_counts()
train[train.text.str.contains('vanquisheth')]
len(train[train.text.str.contains('from')])
wordlist = {}

word_rex = re.compile('[A-Za-z]{2,}')

for i,t in train.iterrows():

    for w in word_rex.findall(t.text):

        if w not in wordlist:

            wordlist[w] = 1

        else:

            wordlist[w] += 1

print("Built wordlist frequencies")            
rare_map = {}

for i,t in train.iterrows():

    fs = []

    for w in word_rex.findall(t.text):

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
pd.Series([len(v) for v in rare_map.values()]).describe()
def find(pt):

    fs = []

    for w in word_rex.findall(pt):

        if w in rare_map.keys():

            fs.append((w, wordlist[w]))

    if len(fs)==0:

        return None

    fs.sort(key=lambda x:x[1])

    for rare_w, _ in fs[:5]: #We'll check up to 5 rare words, just to be safe.

        for t in rare_map[rare_w]:

            if t.text in pt:

                return t
pt = "9]8pVnN4n,DaA6[XNib4K2yVIn[jk[MW0VTo5?J62P?'.0HbpEnter, from one side, LUCIUS, IACHIMO, and  the Roman Army: from the other side, the  British Army, POSTHUMUS LEONATUS following,  like a poor soldier. They march over and go  out. Then enter again, in skirmish, IACHIMO  and POSTHUMUS LEONATUS he vanquisheth and disarmeth IACHIMO, and then leaves himvqlhyWqM4ilXEv]dElTRiO2XBC!)9rl(Iy($HLn'd]ktE6b58y"

find(pt).text
cap_map = {}

cap_rex = re.compile('[A-Z]{5,}')

for i,t in train.iterrows():

    for capw in cap_rex.findall(t.text):

        if capw not in cap_map:

            cap_map[capw] = [t]

        else:

            cap_map[capw].append(t)

print("Built hash table")
pd.Series([len(v) for v in cap_map.values()]).describe()
def find2(pt):

    for cap_w in cap_rex.findall(pt):

        if cap_w in cap_map: #We should only need to check one of the capitalised words

            for t in cap_map[cap_w]:

                if t.text in pt:

                    return t
find2(pt).text