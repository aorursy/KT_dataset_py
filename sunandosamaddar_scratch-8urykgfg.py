import numpy as np

import pandas as pd

print('numpy@' + np.__version__ + '\n' + 'pydata.pandas@' + pd.__version__)
!ls ../input/
train_df = pd.read_csv('/kaggle/input/ciphertext-challenge-iii/train.csv')

test_df = pd.read_csv('/kaggle/input/ciphertext-challenge-iii/test.csv')

sample_sub_df = pd.read_csv('/kaggle/input/ciphertext-challenge-iii/sample_submission.csv')
# Make some room 

pd.set_option('display.max_colwidth',122)
train_df.iloc[:50,:]
test_df.head(10)
sample_sub_df.head()
sample_sub_df.loc[sample_sub_df['ciphertext_id'] == 'ID_55f57ffd0',:]
train_df.shape, test_df.shape
train_df.plaintext_id.nunique()
train_df.index.nunique()
test_df.ciphertext_id.nunique()
sample_sub_df.iloc[0,0] in train_df.plaintext_id
test_df.iloc[0,0] in train_df.plaintext_id
sample_sub_df.loc[0,'ciphertext_id'] in test_df.ciphertext_id 
train_df.info()
train_df.loc[train_df['index'] == 0, :]
train_df.iloc[0,1]
train_df.insert(1,'length_plaintext_id',train_df['plaintext_id'].apply(lambda i: len(i)-3),allow_duplicates=True)

train_df.insert(3,'length_plaintext',train_df['text'].apply(len),allow_duplicates=True)
train_df.iloc[:12,:]
train_df['length_plaintext_id'].unique()
test_df.insert(1,'length_ciphertext_id',test_df['ciphertext_id'].apply(lambda i: len(i)-3),allow_duplicates=True)

test_df.insert(3,'length_ciphertext',test_df['ciphertext'].apply(len),allow_duplicates=True)
test_df.iloc[:24,:]
test_df['length_ciphertext_id'].unique()
bytes('∆PQR',encoding='utf-8')
bytearray(train_df.iloc[7,2],encoding='utf-8')
[ch for ix,ch in enumerate('©2019')]
# Series vars

plaintext_unicode_pts = train_df['text'].apply(lambda string: [ord(ch) for i,ch in enumerate(string)])

ciphertext_unicode_pts = test_df['ciphertext'].apply(lambda string: [ord(ch) for i,ch in enumerate(string)])
import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline

sns.set_style('whitegrid')
def padws(tx):

    while len(tx) % 100 != 0: 

        tx += ' '

    

    return tx



train_df['text'].apply(padws).head()
train_df.insert(4,

               'pad_plaintext',

               train_df['text'].apply(padws),

               allow_duplicates=True)



train_df.insert(5,

               'pad_length_plaintext',

               train_df['length_plaintext'].apply(lambda l: (((l//100)+1)*100) if l%100 != 0 else l),

               allow_duplicates=True)



train_df.head()
test_df.insert(4,

               'pad_length_ciphertext',

               test_df['length_ciphertext'].apply(lambda l: (((l//100)+1)*100) if l%100 != 0 else l),

               allow_duplicates=True)

test_df['pad_length_ciphertext'].head()
np.arange(0, 8000, 500)
f, ax = plt.subplots(nrows=3,ncols=1,figsize=(24, 32))



ax[0].set_title("Freq. distribution of length_plaintext", fontdict={"fontsize": 18})

sns.distplot(train_df['length_plaintext'],

             rug=True,

             kde_kws={

                 "color": "blue",

                 "lw": 2,

                 "label": "KDE"

             },

             hist_kws={

                 "histtype": "bar",

                 "linewidth": 3,

                 "alpha": 1,

                 "color": "lightgreen"

             },

             ax=ax[0])

ax[0].set_xticklabels(labels=ax[0].get_xticks(), fontdict={"fontsize":18})

ax[0].set_yticklabels(labels=ax[0].get_yticks(), fontdict={"fontsize":18})



ax[1].set_title("Freq. distribution of length_ciphertext w/o PADDING", fontdict={"fontsize": 18}) # w/o padding

sns.distplot(test_df['length_ciphertext'],

             rug=True,

             kde_kws={

                 "color": "blue",

                 "lw": 2,

                 "label": "KDE"

             },

             hist_kws={

                 "histtype": "bar",

                 "linewidth": 3,

                 "alpha": 1,

                 "color": "pink"

             },

             ax=ax[1])

ax[1].set_xticklabels(labels=ax[1].get_xticks(), fontdict={"fontsize":18})

ax[1].set_yticklabels(labels=ax[1].get_yticks(), fontdict={"fontsize":18})



ax[2].set_title("Freq. distribution of length_ciphertext w/ PADDING", fontdict={"fontsize": 18}) # w/ padding

sns.distplot(test_df['pad_length_ciphertext'],

             rug=True,

             kde_kws={

                 "color": "blue",

                 "lw": 2,

                 "label": "KDE"

             },

             hist_kws={

                 "histtype": "bar",

                 "linewidth": 3,

                 "alpha": 1,

                 "color": "pink"

             },

             ax=ax[2])

ax[2].set_xticklabels(labels=ax[2].get_xticks(), fontdict={"fontsize":18})

ax[2].set_yticklabels(labels=ax[2].get_yticks(), fontdict={"fontsize":18})
# Normalized distribution of padded ciphertext length counts 

test_df['pad_length_ciphertext'].value_counts(normalize=True)
ciphertexts_cl1 = test_df.loc[test_df['difficulty'] == 1, :].copy()

print(ciphertexts_cl1.shape)

ciphertexts_cl1
ciphertexts_cl1['pad_length_ciphertext'].value_counts(normalize=True)
RS = np.random.RandomState(seed=1224)
sample_cl1 = ciphertexts_cl1.sample(n=5, random_state=RS, axis=0)

sample_cl1
save = pd.Series([ch for i,ch in enumerate(sample_cl1.loc[67731, 'ciphertext'])], dtype=np.object)

save.value_counts()
save.value_counts().index.tolist()
def make_save(index):

    return pd.Series([ch for i,ch in enumerate(sample_cl1.loc[index, 'ciphertext'])], dtype=np.object)



sample_cl1.index.tolist()
f, ax = plt.subplots(nrows=5,ncols=1,figsize=(18, 40))





save = make_save(67731)

vc = save.value_counts(ascending=True).index.tolist()

ax[0].set_title("Freq. distribution of chars in sample", fontdict={"fontsize": 18})

sns.countplot(x=save.values.tolist(), data=save, order=vc, orient='v', palette='PuRd', saturation=0.75, dodge=True, ax=ax[0])

ax[0].set_xticklabels(labels=vc, fontdict={"fontsize":18})

ax[0].set_yticklabels(labels=ax[0].get_yticks(), fontdict={"fontsize":18})



save = make_save(8219)

vc = save.value_counts(ascending=True).index.tolist()

sns.countplot(x=save.values.tolist(), data=save, order=vc, orient='v', palette='PuRd', saturation=0.75, dodge=True, ax=ax[1])

ax[1].set_xticklabels(labels=vc, fontdict={"fontsize":18})

ax[1].set_yticklabels(labels=ax[1].get_yticks(), fontdict={"fontsize":18})



save = make_save(44956)

vc = save.value_counts(ascending=True).index.tolist()

sns.countplot(x=save.values.tolist(), data=save, order=vc, orient='v', palette='PuRd', saturation=0.75, dodge=True, ax=ax[2])

ax[2].set_xticklabels(labels=vc, fontdict={"fontsize":18})

ax[2].set_yticklabels(labels=ax[2].get_yticks(), fontdict={"fontsize":18})



save = make_save(25629)

vc = save.value_counts(ascending=True).index.tolist()

sns.countplot(x=save.values.tolist(), data=save, order=vc, orient='v', palette='PuRd', saturation=0.75, dodge=True, ax=ax[3])

ax[3].set_xticklabels(labels=vc, fontdict={"fontsize":18})

ax[3].set_yticklabels(labels=ax[3].get_yticks(), fontdict={"fontsize":18})



save = make_save(107305)

vc = save.value_counts(ascending=True).index.tolist()

sns.countplot(x=save.values.tolist(), data=save, order=vc, orient='v', palette='PuRd', saturation=0.75, dodge=True, ax=ax[4])

ax[4].set_xticklabels(labels=vc, fontdict={"fontsize":18})

ax[4].set_yticklabels(labels=ax[4].get_yticks(), fontdict={"fontsize":18})
ax
string = sample_cl1.loc[67731, 'ciphertext']

string
from functools import reduce





# len is 100, let's break it into 51-49

output = []

for i,chi in enumerate(string[:51]):

    if i > len(string[51:])-1:

        output.append(chi)

    else:

        for j,chj in enumerate(string[51:]):

            if j == i:

                output.append(chi + chj)

            continue





save = reduce(lambda a,b: ''.join([a, b]), output)

print(save)
assert len(save) == 100

print(save[:51])

print(save[51:])
output_2 = []

for i,chi in enumerate(save[:51]):

    if i > len(save[51:])-1:

        output_2.append(chi)

    else:

        for j,chj in enumerate(save[51:]):

            if j == i:

                output_2.append(chi + chj)

            continue



save = reduce(lambda a,b: ''.join([a, b]), output_2)

print(save)
assert len(save) == 100

print(save[:51])

print(save[51:])
output_3 = []

for i,chi in enumerate(save[:51]):

    if i > len(save[51:])-1:

        output_3.append(chi)

    else:

        for j,chj in enumerate(save[51:]):

            if j == i:

                output_3.append(chi + chj)

            continue



save = reduce(lambda a,b: ''.join([a, b]), output_3)

print(save)
assert len(save) == 100

print(save[:51])

print(save[51:])
output_4 = []

for i,chi in enumerate(save[:51]):

    if i > len(save[51:])-1:

        output_4.append(chi)

    else:

        for j,chj in enumerate(save[51:]):

            if j == i:

                output_4.append(chi + chj)

            continue



save = reduce(lambda a,b: ''.join([a, b]), output_4)

print(save)
output_5 = []

for i,chi in enumerate(save[:51]):

    if i > len(save[51:])-1:

        output_5.append(chi)

    else:

        for j,chj in enumerate(save[51:]):

            if j == i:

                output_5.append(chi + chj)

            continue





save = reduce(lambda a,b: ''.join([a, b]), output_5)

print(save)
sample_cl1.loc[44956,'ciphertext']
np.array([{'a':2,'b':5}, {'c':3,'d':1}])
charspace_cl1 = sorted(set([v for i,v in enumerate(reduce(lambda a,b: '\n'.join([a, b]), ciphertexts_cl1['ciphertext']))]))

charspace_cl1_dict = {}

charspace_cl1_dictrev = {}

for i,v in enumerate(charspace_cl1):

    charspace_cl1_dict[v] = i

    charspace_cl1_dictrev[i] = v

                                                                       #--- this list is for unicode pt values ---#

CHAR_REFERENCE = np.array([charspace_cl1_dict, charspace_cl1_dictrev, [ord(k) for k in charspace_cl1_dict.keys()]])

CHAR_REFERENCE
sample_cl1.loc[67731,'ciphertext']
test_cip = sample_cl1.loc[67731,'ciphertext']

test_cip_arrfy = [vi for i,vi in enumerate(test_cip)]



# check digrams

arr, tuparr = [], []

for i,v in enumerate(test_cip_arrfy):

    if i == len(test_cip_arrfy)-2:

        break

    

    s = test_cip[i] + test_cip[i+1]

    c = test_cip.count(s,0,-1)

    if c > 0:

        arr.append(c)

        tuparr.append((s,c))

    

max(arr)
digram_freqs = pd.Series(tuparr)

top_dgms = []



for t in digram_freqs:

    if t[1] == 2:

        top_dgms.append(t[0])

        

set(top_dgms)
test_cip
save, save_b = [], [] 

map_cip_sample = ""





for i,v in enumerate(sample_cl1.loc[67731,'ciphertext']):

    save.append(charspace_cl1_dict[v])

print(save)

    

def deshift_with_wrap(v, bias=-4):    

    if v in (list(range(0, 21)) + [46, 47]): return v # skip punctuation and digits 

    

    if bias < 0:

        if v-bias > 73: return v-bias - 74

        return v-bias

    else:

        if v-bias < 0: return v-bias + 74

        if v-bias > 73: return v-bias - 74

        return v-bias



save_b = list(map(deshift_with_wrap, save))

print(save_b)



for i,v in enumerate(save_b):

    map_cip_sample += charspace_cl1_dictrev[v]

    

    

map_cip_sample
'abbbM: Ntmj wwmh wwqm sp m$vm nr jpx Fpx!wet'
all_occurr_cip = pd.Series([v for i,v in enumerate(reduce(lambda a,b: '\n'.join([a, b]), ciphertexts_cl1['ciphertext']))], dtype=np.object)



# sample from train

sample_train = train_df.loc[train_df['pad_length_plaintext'] == 100, 'text'].sample(n=ciphertexts_cl1.shape[0], random_state=RS, axis=0)

sample_occurr_plain = pd.Series([v for i,v in enumerate(reduce(lambda a,b: '\n'.join([a, b]), sample_train))], dtype=np.object)
charspace_trainset = sorted(set([v for i,v in enumerate(reduce(lambda a,b: '\n'.join([a, b]), train_df['text']))]))

f, ax = plt.subplots(nrows=2,ncols=1,figsize=(18, 18))





vc = all_occurr_cip.value_counts(normalize=True, ascending=True).index.tolist()

ax[0].set_title("Freq. distribution of chars in sample", fontdict={"fontsize": 18})

sns.countplot(x=all_occurr_cip.values.tolist(), data=all_occurr_cip, order=vc,orient='v', palette='PuRd', saturation=0.75, dodge=True, ax=ax[0])

ax[0].set_xticklabels(labels=vc, fontdict={"fontsize":18})

ax[0].set_yticklabels(labels=ax[0].get_yticks(), fontdict={"fontsize":18})



vc = sample_occurr_plain.value_counts(normalize=True, ascending=True).index.tolist()

ax[1].set_title("Freq. distribution of chars in all <=100 charlength plaintexts", fontdict={"fontsize": 18})

sns.countplot(x=sample_occurr_plain.values.tolist(), data=sample_occurr_plain, order=vc, orient='v', palette='PuRd', saturation=0.75, dodge=True, ax=ax[1])

ax[1].set_xticklabels(labels=vc, fontdict={"fontsize":18})

ax[1].set_yticklabels(labels=ax[1].get_yticks(), fontdict={"fontsize":18})
# Digram frequency graph 

sample_concat_plain = reduce(lambda a,b: '\n'.join([a,b]), sample_train)

arr, tuparr = [], []

for i,v in enumerate(sample_concat_plain):

    if i == len(sample_concat_plain)-2:

        break

    

    s = sample_concat_plain[i] + sample_concat_plain[i+1]

    c = sample_concat_plain.count(s,0,-1)

    if c > 0:

        arr.append(c)

        tuparr.append((s,c))

    

print(max(arr))
tuparr
top_digram_freqs = pd.DataFrame(data={

    #"main": tuparr,

    "chars": [t[0] for t in tuparr],

    "counts": [t[1] for t in tuparr]

})



assert top_digram_freqs.shape[0] == len(tuparr)
top_digram_freqs.drop_duplicates(inplace=True)

top_digram_freqs.sort_values(by=['counts'], axis=0, ascending=False, inplace=True)
top_digram_freqs.iloc[:50, 0]
f, ax = plt.subplots(nrows=1,ncols=1,figsize=(18, 9))



vc = top_digram_freqs.iloc[:50, 0].copy()

ax.set_title("Freq. distribution of top digrams in all <=100 charlength plaintexts", fontdict={"fontsize": 18})

sns.barplot(x='chars', y='counts', data=top_digram_freqs.iloc[:50,:], order=vc, orient='v', palette='pink', saturation=0.75, dodge=True, ax=ax)

ax.set_xticklabels(labels=vc, fontdict={"fontsize":18}, rotation=30)

ax.set_yticklabels(labels=ax.get_yticks(), fontdict={"fontsize":18})
%%time

'asdf'+'<BEGIN>'+'qwer'
%%time

'<BEGIN>'.join(['asdf', 'qwer'])
# Digram frequency graph 

sample_concat_cip = reduce(lambda a,b: '\n'.join([a, b]), ciphertexts_cl1['ciphertext'])

arr, tuparr = [], []

for i,v in enumerate(sample_concat_cip):

    if i == len(sample_concat_cip)-2:

        break

    

    s = sample_concat_cip[i] + sample_concat_cip[i+1]

    c = sample_concat_cip.count(s,0,-1)

    if c > 0:

        arr.append(c)

        tuparr.append((s,c))

    

print(max(arr))
top_digram_freqs = pd.DataFrame(data={

    #"main": tuparr,

    "chars": [t[0] for t in tuparr],

    "counts": [t[1] for t in tuparr]

})

assert top_digram_freqs.shape[0] == len(tuparr)



top_digram_freqs.drop_duplicates(inplace=True)

top_digram_freqs.sort_values(by=['counts'], axis=0, ascending=False, inplace=True)





f, ax = plt.subplots(nrows=1,ncols=1,figsize=(18, 9))



vc = top_digram_freqs.iloc[:50, 0].copy()

ax.set_title("Freq. distribution of top digrams in all ciphertexts", fontdict={"fontsize": 18})

sns.barplot(x='chars', y='counts', data=top_digram_freqs.iloc[:50,:], order=vc, orient='v', palette='pink', saturation=0.75, dodge=True, ax=ax)

ax.set_xticklabels(labels=vc, fontdict={"fontsize":18}, rotation=30)

ax.set_yticklabels(labels=ax.get_yticks(), fontdict={"fontsize":18})
# Replacing 's' with 'e'

#           'd' with 't' 

test_sent = sample_cl1.loc[44956,'ciphertext']

print(test_sent)

print('cipher 1 predictions demo')

reduce(lambda a,b: ''.join([a,b]),['h' if v == 's' else #ok

                                   #'e' if v == 'd' else

                                   #'h' if v == 'f' else

                                   #'o' if v == 'j' else

                                   #'e' if v == 'i' else

                                   'a' if v == 'x' else #ok

                                   'b' if v == 'l' else #ok

                                   'm' if v == 'p' else #ok

                                   #'s' if v == 'e' else

                                   'r' if v == 't' else #ok

                                   'A' if v == 'B' else #ok

                                   #'r' if v == 'u' else 

                                   's' if v == 'w' else #ok

                                   #'r' if v == 'y' else 

                                   #'v' if v == 'r' else 

                                   v for i,v in enumerate(test_sent)])
train_df.columns
crack_1 = train_df['text'].str.find("Abraham's")

crack_1.value_counts()
crack_1.loc[crack_1 == 28].index
train_df.iloc[52854, :]
sample_cl1.index.tolist()
# Trying another with Abraham's blessing

test_sent_2 = sample_cl1.loc[25629,'ciphertext']

print(test_sent_2)

print('cipher 1 predictions demo')

reduce(lambda a,b: ''.join([a,b]),['h' if v == 's' else #ok

                                   'e' if v == 'd' else #ok

                                   't' if v == 'f' else #ok

                                   'h' if v == 'w' else #ok

                                   'h' if v == 'g' else #ok

                                   'a' if v == 'x' else #ok

                                   'b' if v == 'l' else #ok

                                   'e' if v == 'p' else #ok

                                   #'o' if v == 'm' else

                                   'r' if v == 't' else #ok

                                   'A' if v == 'B' else #ok

                                   's' if v == 'i' else #ok

                                   'i' if v == 'm' else #ok

                                   #'f' if v == 'b' else 

                                   v for i,v in enumerate(test_sent_2)])
crack_1 = train_df['text'].str.find("he '")

crack_1.value_counts()
train_df.loc[train_df.loc[crack_1 != -1, :].index.tolist(), 'text']



# Matches with index 79849
train_df.loc[79849, 'text']
test_sent_3 = sample_cl1.loc[107305,'ciphertext']

print(test_sent_3)

print('cipher 1 predictions demo')

reduce(lambda a,b: ''.join([a,b]),['h' if v == 's' else #ok

                                   'e' if v == 'd' else #ok

                                   't' if v == 'f' else #ok

                                   'h' if v == 'w' else #ok

                                   'h' if v == 'g' else #ok

                                   'a' if v == 'x' else #ok

                                   'b' if v == 'l' else #ok

                                   'e' if v == 'p' else #ok

                                   'h' if v == 'o' else #ok

                                   'r' if v == 't' else #ok

                                   'A' if v == 'B' else #ok

                                   's' if v == 'i' else #ok

                                   'i' if v == 'm' else #ok

                                   't' if v == 'j' else #ok 

                                   v for i,v in enumerate(test_sent_3)])
crack_1 = train_df['text'].str.find("a hair,")

crack_1.value_counts()
train_df.loc[train_df.loc[crack_1 != -1].index.tolist(), :]



# Matches with 80856
train_df.loc[80856, 'text']
sample_sub_df.head()
test_df.loc[[44956,25629,107305], 'ciphertext_id'].values.tolist()
submission = pd.DataFrame(data={

    'ciphertext_id': test_df.loc[[44956,25629,107305], 'ciphertext_id'].values.tolist(),

    'index': [52854,79849,80856]

})



submission
# Running out of time ...