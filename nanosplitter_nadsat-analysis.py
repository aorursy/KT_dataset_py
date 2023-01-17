import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

def data_prep(raw):
    out_y = raw.values[...,0,]
    batch_size = raw.shape[0]
    out_x = raw.values[...,1,]
    return list(out_x), list(out_y)
file = '../input/nadsat vocab.csv'
raw_data = pd.read_csv(file)
english, nadsat = data_prep(raw_data)

raw_data['word'].value_counts().head(30).plot.bar()
raw_data['definition'].value_counts().head(30).plot.bar()
#Word Object
class Word:
    def __init__(self, word, count):
        self.word = word
        self.count = count
    def __repr__(self):
        return "{"+str(self.word)+ ", " +str(self.count)+"}"
    def __eq__(self, obj):
        return self.word == obj.word and self.count == obj.count
    def __hash__(self):
        return hash(self.word)

#Getting all prefixes of word
def gas(input_string):
  length = len(input_string)
  return [input_string[:i+1] for i in range(length)]

freq = []
subs = []

#Sorting prefixes into frequency list
for word in nadsat:
    for s in gas(word):
        if len(s) == 3:
            subs.append(s)
for s in subs:
    freq.append(Word(s, subs.count(s)))
freq = sorted(set(freq), key = lambda x: x.count)[::-1]

perm = []
temp = []

#Connecting frequency list with english definition list
for s in freq:
    temp = []
    temp.append(s.word)
    for i in range(len(nadsat)):
        if s.word in gas(nadsat[i]):
            temp.append(english[i])
    perm.append(temp)

print("Prefix ", " English counterparts to Nadsat words with prefix")
#Printing results
for i in range(10):
    res = ""
    res += perm[i][0] + ":     "
    for w in range(len(perm[i])):
        if w != 0:
            res += perm[i][w] + " | "
    print(res)
        


#Word Object
class Word:
    def __init__(self, word, count):
        self.word = word
        self.count = count
    def __repr__(self):
        return "{"+str(self.word)+ ", " +str(self.count)+"}"
    def __eq__(self, obj):
        return self.word == obj.word and self.count == obj.count
    def __hash__(self):
        return hash(self.word)

#Getting all prefixes of word
def gas(input_string):
  length = len(input_string)
  return [input_string[i:] for i in range(length)]

freq = []
subs = []

#Sorting prefixes into frequency list
for word in nadsat:
    for s in gas(word):
        if len(s) == 3:
            subs.append(s)
for s in subs:
    freq.append(Word(s, subs.count(s)))
freq = sorted(set(freq), key = lambda x: x.count)[::-1]

perm = []
temp = []

#Connecting frequency list with english definition list
for s in freq:
    temp = []
    temp.append(s.word)
    for i in range(len(nadsat)):
        if s.word in gas(nadsat[i]):
            temp.append(english[i])
    perm.append(temp)

print("Suffix ", " English counterparts to Nadsat words with suffix")
#Printing results
for i in range(10):
    res = ""
    res += perm[i][0] + ":     "
    for w in range(len(perm[i])):
        if w != 0:
            res += perm[i][w] + " | "
    print(res)
        


def v_count(string):
    c = 0
    for i in string:
        if i in "aeiouAEIOU":
            c += 1
    return c

def c_count(string):
    c = 0
    for i in string:
        if i not in "aeiouAEIOU":
            c += 1
    return c

english_c = 0
english_v = 0

for i in english:
    english_c += c_count(i)
    english_v += v_count(i)

nadsat_v = 0
nadsat_c = 0

for i in nadsat:
    nadsat_c += c_count(i)
    nadsat_v += v_count(i)

print("Nadsat consonant to vowel ratio:", round(nadsat_c / nadsat_v, 2))
print("English consonant to vowel ratio:", round(english_c / english_v, 2))
nadsat_len = 0
english_len = 0

for i in english:
    english_len += len(i)

for i in nadsat:
    nadsat_len += len(i)
    
print("Average Nadsat word length:", round(nadsat_len / len(nadsat), 2))
print("Average English word length:", round(english_len / len(english), 2))