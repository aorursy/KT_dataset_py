import pandas as pd

import numpy as np

from tqdm.auto import tqdm

import matplotlib.pyplot as plt

from pprint import pprint

import seaborn as sns

from collections import Counter

from itertools import cycle, product

from string import ascii_lowercase, ascii_uppercase

from flashtext import KeywordProcessor

import base64



pd.set_option('display.max_rows', 5000)

pd.set_option('display.max_columns', 5000)

pd.set_option('display.max_colwidth', 5000)

pd.set_option('display.width', 5000)



train = pd.read_csv("../input/train.csv", index_col='index', usecols=['index', 'text'])

test = pd.read_csv('../input/test.csv', index_col='ciphertext_id')

sub = pd.read_csv('../input/sample_submission.csv', index_col='ciphertext_id')

train['length'] = train.text.apply(len)

test['length'] = test.ciphertext.apply(len)



def encode_level_1(text, i=0, key='pyle'):

    key = [ord(x) - 97 for x in key]

    def substitute(char):

        nonlocal i

        if char in ascii_lowercase and char != 'z':

            char = chr((ord(char) - 97 + key[i]) % 25 + 97)

            i = (i + 1) % len(key)

        if char in ascii_uppercase and char != 'Z':

            char = chr((ord(char) - 65 + key[i]) % 25 + 65)

            i = (i + 1) % len(key)

        return char

    return ''.join([substitute(x) for x in text])



def rail_pattern(n):

    r = list(range(n+1))

    return cycle(r + r[-2:0:-1])



def encode_level_2(text, rails=20):

    p = rail_pattern(rails)

    result = sorted(text, key=lambda i: next(p))

    return ''.join(result)



def decode_level_1(text, i=0, key='pyle'):

    key = [ord(x) - 97 for x in key]

    def substitute(char):

        nonlocal i

        if char in ascii_lowercase and char != 'z':

            char = chr((ord(char) - 97 - key[i]) % 25 + 97)

            i = (i + 1) % len(key)

        if char in ascii_uppercase and char != 'Z':

            char = chr((ord(char) - 65 - key[i]) % 25 + 65)

            i = (i + 1) % len(key)

        return char

    return ''.join([substitute(x) for x in text])



def decode_level_2(text, rails=20):

    p = rail_pattern(rails)

    indexes = sorted(range(len(text)), key=lambda i: next(p))

    result = [''] * len(text)

    for i, c in zip(indexes, text):

        result[i] = c

    return ''.join(result)



from urllib.request import urlopen

with urlopen("https://www.gutenberg.org/files/46464/46464-0.txt") as key_file:

    key_level3 = key_file.read().decode('utf-8').replace('\r', ' ').replace('\n', ' ')

    

def decode_level_3(text, key=key_level3):

    return ''.join([key_level3[int(n)] for n in text.split(" ")])



test.loc[test["difficulty"] == 1,"text"] = test.loc[test["difficulty"] == 1,"ciphertext"].map(lambda x: decode_level_1(x))

test.loc[test["difficulty"] == 2,"text"] = test.loc[test["difficulty"] == 2,"ciphertext"].map(lambda x: decode_level_1(decode_level_2(x)))

test.loc[test["difficulty"] == 3,"text"] = test.loc[test["difficulty"] == 3,"ciphertext"].map(lambda x: decode_level_1(decode_level_2(decode_level_3(x))))



test1 = test[test["difficulty"] == 1].copy()

test2 = test[test["difficulty"] == 2].copy()

test3 = test[test["difficulty"] == 3].copy()

test4 = test[test["difficulty"] == 4].copy()



print(encode_level_2(encode_level_1(train["text"][1])))
keyword_processor = KeywordProcessor(case_sensitive=True)

keyword_processor.set_non_word_boundaries(set())



for index, text, length in tqdm(train.itertuples(), total=train.shape[0], mininterval=1):

    if len(text) < 3 or text == 'So.':

        continue

    keyword_processor.add_keyword(text, index)

print(len(keyword_processor))



def good_match(match, text):

    d = (len(text) - len(match)) // 2

    return match == text[d:d+len(match)]



matched, unmatched = 0, 0

def match_row(text):

    global matched, unmatched

    try:

        if text != text:

            return 0

        matches0 = keyword_processor.extract_keywords(text)

        matches = [x for x in matches0 if good_match(train.loc[x]['text'], text)]

        if len(matches) == 1:

            matched += 1

            return matches[0]

        else:

            print(text, matches0, [train.loc[x]['text'] for x in matches0])

            unmatched += 1

            return 0

    except KeyError:

        unmatched += 1

        print(text)

        return 0



test['result'] = test.text.map(match_row)

sub["index"] = test.loc[sub.index]['result']

print(f"Matched {matched}   Unmatched {unmatched}")
sub123 = sub.copy()
level123_train_index = list(sub[sub["index"] > 0]["index"])

print(len(level123_train_index))

train4 = train[~train.index.isin(level123_train_index)].copy()

print(len(train4))
def get_distribution(texts):

    return pd.Series([sorted(pd.Series([x[i] for x in texts if len(x) > i]).drop_duplicates().values) for i in range(1200)])



test4_decoded = [base64.b64decode(x.encode()) for x in test4.ciphertext]

distribution = get_distribution([x.encode() for x in test3.ciphertext]).to_frame('test3')

distribution['test4'] = get_distribution(test4_decoded)

print(distribution)
def distribution_matches(d1, d2):

    return all(y in d1 for y in d2)



def distribution_matches_ij(i, j):

    return distribution_matches([32, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57], [x^j for x in distribution['test4'][i]])



distribution['key_index'] = [next((j for j in range(256) if distribution_matches_ij(i, j)), -1) for i in range(1200)]

print(distribution[distribution['key_index'] == -1])

distribution['key_index2'] = [next(((255-j) for j in range(256) if distribution_matches_ij(i, (255-j))), -1) for i in range(1200)]

print(distribution[distribution['key_index2'] == -1])
key4 = list(distribution['key_index'])

key42 = list(distribution['key_index2'])

print([(x,y) for x,y in zip(key4,key42) if x != y])



def decode_level_4(text, key=key42):

    return ''.join([chr(a^b) for a,b in zip(base64.b64decode(text.encode()),key)])



competition_title = "BRBTvl0LNstxQLyxulCEEq1czSFje0Z6iajczo6ktGmitTE="

decode_level_1(decode_level_2(decode_level_3(decode_level_4(competition_title))))
def decode_level_3(text, key=key_level3):

    return ''.join([key_level3[int(n)] for n in text.split(" ") if n != ''])



def pad_str(s, special_char = '*'):

    nb = len(s)

    nb_round = ((nb + 99) // 100) * 100

    nb_left = (nb_round - nb) // 2

    nb_right = nb_round - nb - nb_left

    

    left_s = ''.join([special_char] * nb_left)

    right_s = ''.join([special_char] * nb_right)

    return left_s + s + right_s
test.loc[test["difficulty"] == 4,"text3"] = test.loc[test["difficulty"] == 4,"ciphertext"].map(lambda x: decode_level_4(x))

test.loc[test["difficulty"] == 4,"text2"] = test.loc[test["difficulty"] == 4,"text3"].map(lambda x: decode_level_3(x))

test.loc[test["difficulty"] == 4,"text1"] = test.loc[test["difficulty"] == 4,"text2"].map(lambda x: decode_level_2(x))

test.loc[test["difficulty"] == 4,"text"] = test.loc[test["difficulty"] == 4,"text1"].map(lambda x: decode_level_1(x))

test.loc[test["difficulty"] == 4,"nb"] = test.loc[test["difficulty"] == 4,"text3"].apply(lambda x: len(x.split(' ')))

test.loc[test["difficulty"] == 4,"len"] = test.loc[test["difficulty"] == 4,"ciphertext"].apply(lambda x: len(x))
a = '1053 45857 35608 18490 42966 29015 603 43218 6934 37471 43224 44901 34563 31567 43101 12421 7494 28531 46902 13301 45969 32885 34400 44280 43145 26701 42951 22568 42985 40528 1092 44309 22643 5187 33164 2207 48614 19535 42991 45453 27045 42533 22190 15410 1914 25495 24705 1561 18313 28416 45709 23929 43099 43117 48113 480 42893 44280 44240 20578 25706 9948 47127 42985 31261 28384 10513 5492 48613 49372 12108 39746 32253 31320 39684 32942 21377 31648 43299 4643 14420 44363 44096 46100 46603 42912 13348 38771 48937 33534 31333 43067 2460 8980 20432 18020 16122 50198 36242 11252 1131 43072 12660 17089 2816 42985 30094 38425 32900 44474 43147 9041 49153 36245 8020 2401 32940 37916 4842 25172 1571 42168 30302 2950 12377 31337 36218 45690 42985 19776 655 22716 50019 27443 50282 43191 24707 35227 42970 11107 46392 39666 2671 14644 14817 49200 47899 25008 47186 43145 43914 41506 39404 35972 3118 44293 5293 6591 40850 27894 16878 36505 12814 14028 36334 26413 21894 24270 23091 48488 962 19153 6576 22542 22925 34911 35072 26603 11578 17163 48375 18672 35086 20746 38125 44898 13950 4114 3373 33672 6732 16900 4464 17951 11855 27465 3317 20227 36175 41895'

decode_level_3(a)
print([encode_level_2(encode_level_1(pad_str(x))) for x in train.text if 'BUCKINGHAM: Cornets. Enter KING HENRY' in x])

print([x for x in test.itertuples() if 'BUCKINGHAM: Cornets. Enter KING HENRY' in x.text])
test.loc[test["difficulty"] == 4,"text"] = test.loc[test["difficulty"] == 4,"ciphertext"].map(lambda x: decode_level_1(decode_level_2(decode_level_3(decode_level_4(x)))))

test.loc[test["difficulty"] == 4,"text"]
def letter_frequency_stats(texts):

    memo = Counter("".join([x for x in texts]))

    mstats = pd.DataFrame([[x[0], x[1]] for x in memo.items()], columns=["Letter", "Frequency"])

    return mstats.sort_values(by='Frequency', ascending=False)



letterStats = letter_frequency_stats([' '.join([str(x) for x in key42])])



plt.figure(figsize=(30, 4))



plt.subplot(1, 1, 1)

print(len(letterStats))

plot_series = np.array(range(len(letterStats))) + 0.5

plt.bar(plot_series, letterStats['Frequency'].values)

plt.xticks(plot_series, letterStats['Letter'].values)



# plt.savefig("count.png")

plt.show()
rare_symbols = letterStats.loc[letterStats["Frequency"] < 100]['Letter'].values

rare_occurances = [x for x in train["text"] if np.any(np.isin(rare_symbols, [y for y in x]))]

print(rare_symbols)

pprint(rare_occurances)
def word_frequency_stats(texts):

    memo = Counter(" ".join(texts).split(" "))

    mstats = pd.DataFrame([[x[0], x[1]] for x in memo.items() if len(x[0]) > 0], columns=["Word", "Frequency"])

    return mstats.sort_values(by='Frequency', ascending=False)



cipherStats = word_frequency_stats([' '.join([str(x) for x in key4])])

cipherStats['Num'] = cipherStats['Word'].astype(int)

print(len(cipherStats))



plt.figure(figsize=(30, 30))

nsize = 127

plot_series = np.array(range(nsize)) + 0.5



for i in range(2):

    plt.subplot(10, 1, i+1)

    plt.bar(plot_series, cipherStats['Frequency'].values[nsize*i:nsize*(i+1):1])

    #num_ch_mapping.get(int(x), '*')

    plt.xticks(plot_series, [x for x in cipherStats['Num'].values[nsize*i:nsize*(i+1):1]])



cs = cipherStats.sort_values("Num", ascending=True).reset_index()

print([hex(x) for x in cs.loc[[0,cs.shape[0]-1]]["Num"]])

# plt.savefig("count.png")

plt.show()
test3["nb"] = test3["ciphertext"].apply(lambda x: len(x.split(" ")))

encoded1 = [int(x) for x in test3.sort_values("nb", ascending=False)["ciphertext"].values[0].split(" ")]

decoded1 = '**************' + train.sort_values("length", ascending=False)["text"].values[2] + '***************'

recoded1 = encode_level_2(encode_level_1(decoded1, 3))

print(encoded1)

print(decoded1)

print(is_correct_mapping(recoded1, test3.sort_values("nb", ascending=False)["ciphertext"].values[0]))

list1 = [x for x in zip(recoded1, encoded1, range(len(encoded1)))]

dict1 = {}

x0 = list1[len(list1)-1][0]

for x, y, z in list1:

    if x != '*':

        dict1.setdefault(y, []).append(x)

    x0 = x

# list1.sort(key = lambda x: x[1])

pprint({y: x for y, x in dict1.items()})
sub.to_csv('submit-level-3.csv')