import numpy as np

import pandas as pd

from collections import Counter



train = pd.read_csv('../input/training.csv')

test = pd.read_csv('../input/test.csv')

sub = pd.read_csv('../input/sample_submission.csv')
for difficulty in sorted(test['difficulty'].unique()):

    print('Difficulty: ', difficulty)

    sample = test[test['difficulty']==difficulty]

    print(len(sample), sample['ciphertext'].iloc[0][:50])
traind = Counter(' '.join(train['text'].astype(str).values))

traind = ''.join([c for c, v in traind.most_common(85)])

print(repr(traind))



test1 = test[test['difficulty']==1].reset_index(drop=True)

testd = Counter(' '.join(test1['ciphertext'].astype(str).values))

testd = ''.join([c for c, v in testd.most_common(85)])

print(repr(testd))



#Solution 1 -> Apply to difficulty 2

#Some quick adjustments based on lookups. Not really cryptography but helps to move forward :)

traind = ' etaoisnrhlducmfygwpb.v,kI\'TA"-SEBMxCHDjW)(RLONPGF!Jzq01?KVY:9U2*/3;58476ZQX%$}#@={[]'

testd =  '7lx4v!o2Q[O=y,CzV:}dFX#(Wak/qbne *JAmKp{fc6DGZj\'Tg9"YHS]Ei5)8h1MINwP@s?U3;0%$-rLuBRt.'



SubEnc_ = str.maketrans(testd, traind)



test1['text'] = test1['ciphertext'].map(lambda x: str(x).translate(SubEnc_)).values

test1.head()
def test_contains_text(s):

    val = ''

    s = str(s)

    try:

        val = test1[test1['text'].str.contains(s, regex=False)]['ciphertext_id'].iloc[0]

    except: pass

    return val



train['ciphertext_id'] = train['text'].map(lambda x: test_contains_text(x))

test = pd.merge(test, train[['ciphertext_id', 'index']], how='left', on='ciphertext_id')

test['index'].fillna(sub['index'], inplace=True)

test['index'] = test['index'].astype(int)

test[['ciphertext_id', 'index']].to_csv('submission.csv', index=False)