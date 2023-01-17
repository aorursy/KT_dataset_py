# The package, please to duplicate



import numpy as np

import pandas as pd

from collections import Counter

from tqdm import tqdm_notebook as tqdm
%%time

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

sub_df = pd.read_csv('../input/sample_submission.csv')
train_df.head()
test_df.head()
sub_df.head()
train = train_df.copy()

print(len(train))

train.head()
test = pd.merge(sub_df, test_df, on = ['ciphertext_id'])

print(len(test))

test.head()
test.tail()
test.difficulty.value_counts()
train['word_list'] = train.text.str.split()

train['word_num'] = train['word_list'].map(len)

train.head()
ciph_1 = test[test.difficulty == 1]

ciph_1.set_index('ciphertext_id',inplace=True)

ciph_1.drop(columns = 'difficulty')



ciph_2 = test[test.difficulty == 2]

ciph_2.set_index('ciphertext_id',inplace=True)

ciph_2.drop(columns = 'difficulty')



ciph_3 = test[test.difficulty == 3]

ciph_3.set_index('ciphertext_id',inplace=True)

ciph_3.drop(columns = 'difficulty')



ciph_4 = test[test.difficulty == 4]

ciph_4.set_index('ciphertext_id',inplace=True)

ciph_4.drop(columns = 'difficulty')
ciph_1['ciphertextlist'] = ciph_1.ciphertext.str.split()



ciph_2['ciphertextlist'] = ciph_2.ciphertext.str.split()



ciph_3['ciphertextlist'] = ciph_3.ciphertext.str.split()



ciph_4['ciphertextlist'] = ciph_4.ciphertext.str.split()
train['word_num'].value_counts()
ciph_1['length'] = ciph_1.ciphertextlist.map(len)

print(ciph_1.length.value_counts())



ciph_2['length'] = ciph_2.ciphertextlist.map(len)

print(ciph_2.length.value_counts())



ciph_3['length'] = ciph_3.ciphertextlist.map(len)

print(ciph_3.length.value_counts())



ciph_4['length'] = ciph_4.ciphertextlist.map(len)

print(ciph_4.length.value_counts())
train.head()
test.head()
ciph_1.head()
ciph_2.head()
ciph_3.head()
ciph_4.head()
#Solution 1

traind = ' etaoisnrhlducmfygwpb.v,kI\'TA"-SEBMxCHDjW)(RLONPGF!Jzq01?KVY:9U2*/3;58476ZQX%$}#@={[]'

testd =  '7lx4v!o2Q[O=y,CzV:}dFX#(Wak/qbne *JAmKp{fc6DGZj\'Tg9"YHS]Ei5)8h1MINwP@s?U3;0%$-rLuBRt.'

SubEnc_ = str.maketrans(testd, traind)



#Apply to all

test['ciphertext'] = test['ciphertext'].map(lambda x: str(x).translate(SubEnc_)).values
for difficulty in sorted(test['difficulty'].unique()):

    print('Difficulty: ', difficulty)

    sample = test[test['difficulty']==difficulty]

    print(len(sample), sample['ciphertext'].iloc[0][:50])
traind = Counter(' '.join(train['text'].astype(str).values).split(' ')).most_common(400)

traind = [(w, c) for w, c in traind if len(w)==6]

print(repr(traind))



test2 = test[test['difficulty']==2].reset_index(drop=True)

testd = Counter(' '.join(test2['ciphertext'].astype(str).values).split(' ')).most_common(400)

testd = [(w, c) for w, c in testd if len(w)==6]

print('*********************************************')

print(repr(testd))
t1_ = train[((train['text'].str.contains('should')))]['text'].values[0]

t2_ = test[((test['difficulty']==2))]['ciphertext'].values[0]



for i in range(0, len(t1_), 50):

    print(t1_[i:i+50])

    print(t2_[i:i+50])

    if i > 100:

        break
possible =  ' !"#$%\'()*,-./0123456789:;=?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz{}'

nosub = ''

common = {}



def get_common(t1_, t2_):

    global nosub

    global common

    global possible

    

    for i in range(len(t1_)):

        if str(t1_[i]) in common:

            common[str(t1_[i])]['Subs'].append(str(t2_[i]))

            common[str(t1_[i])]['Subs'] = list(set(common[str(t1_[i])]['Subs']))

            common[str(t1_[i])]['Count'] += 1

        else:

            common[str(t1_[i])] = {'Subs':[str(t2_[i])], 'Count':1}



    for k in sorted(common):

        if common[k]['Count'] > 2 and len(common[k]['Subs'])==1:

            print(k, common[k])

            nosub += k

            nosub = ''.join(sorted(set(nosub)))



    a = ''

    for i in range(len(possible)):

        if possible[i] in list(nosub):

            a += possible[i]

        else:

            a += '*'

    print(a)

    

get_common(t1_, t2_)
%%time

nosub = ' !$%(,1456:BEJPQTUVXYZ]acdghijnvw'

train['len'] = train['text'].map(lambda x: len(str(x)))

test['len'] = test['ciphertext'].map(lambda x: len(str(x)))

i_ = 10

for i in tqdm(range(len(nosub))):

    train['nosub_'+str(nosub[i])] = train['text'].map(lambda x: len([v for v in str(x) if str(v)==str(nosub[i])]))

    train['nosub_'+str(nosub[i])] = train['nosub_'+str(nosub[i])].map(lambda x: x + abs(i_ - (x % i_) if (x % i_) > 0 else x + i_)) #Add some wiggle room

    test['nosub_'+str(nosub[i])] = test['ciphertext'].map(lambda x: len([v for v in str(x) if str(v)==str(nosub[i])]))

    test['nosub_'+str(nosub[i])] = test['nosub_'+str(nosub[i])].map(lambda x: x + abs(i_ - (x % i_) if (x % i_) > 0 else x + i_)) #Add some wiggle room
train.rename(columns={'len': 'true_len'}, inplace=True)

train['len'] = train['true_len'].map(lambda x: x + abs(100 - (x % 100) if (x % 100) > 0 else x))

test_sol2 = pd.merge(test, train, how='inner', on=['nosub_ ',

       'nosub_!', 'nosub_$', 'nosub_%', 'nosub_(', 'nosub_,', 'nosub_1',

       'nosub_4', 'nosub_5', 'nosub_6', 'nosub_:', 'nosub_B', 'nosub_E',

       'nosub_J', 'nosub_P', 'nosub_Q', 'nosub_T', 'nosub_U', 'nosub_V',

       'nosub_X', 'nosub_Y', 'nosub_Z', 'nosub_]', 'nosub_a', 'nosub_c',

       'nosub_d', 'nosub_g', 'nosub_h', 'nosub_i', 'nosub_j', 'nosub_n',

       'nosub_v', 'nosub_w', 'len'])

print(len(test_sol2))

test_sol2.drop_duplicates(subset=['ciphertext_id_x'], keep='first', inplace=True)

#test_sol2 = test_sol2[test_sol2['difficulty']== 2] #Lets let others ride for now

test_sol2['start'] = ((test_sol2['len'] - test_sol2['true_len']) / 2).astype(int)

test_sol2['end'] = (((test_sol2['len'] - test_sol2['true_len']) / 2) + 0.5).astype(int)

print(len(test_sol2))

# solution1 = pd.read_csv('../input/the-crypto-keeper/submission.csv')

test_sol2_ = test_sol2[['ciphertext_id_x', 'index']].rename(columns={'ciphertext_id_x': 'ciphertext_id', 'index': 'index2'})

solution2 = pd.merge(test_sol2_, test_sol2_, how='left', on=['ciphertext_id'])

solution2['index'] = solution2.apply(lambda r: r['index2'] if r['index']==0 else r['index'], axis=1).fillna(0).astype(int)

solution2[['ciphertext_id', 'index']].to_csv('submission.csv', index=False)