# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
tr = pd.read_csv('../input/train.csv')
te = pd.read_csv('../input/test.csv')
from nltk.corpus import stopwords
SCALE = 0.3627
def word_match_share(x):
    '''
    The much-loved word_match_share feature.

    Args:
        x: source data with question1/2
        
    Returns:
        word_match_share as a pandas Series
    '''
    stops = set(stopwords.words('english'))
    q1 = x.question1.fillna(' ').str.lower().str.split()
    q2 = x.question2.fillna(' ').str.lower().str.split()
    q1 = q1.map(lambda l : set(l) - stops)
    q2 = q2.map(lambda l : set(l) - stops)
    q = pd.DataFrame({'q1':q1, 'q2':q2})
    q['len_inter'] = q.apply(lambda row : len(row['q1'] & row['q2']), axis=1)
    q['len_tot'] = q.q1.map(len) + q.q2.map(len)
    return (2 * q.len_inter / q.len_tot).fillna(0)
def bin_model(tr, te, bins=100, vpos=1, vss=3):
    '''
    Runs a Pandas table model using the word_match_share feature.
    
    Args:
        tr: pandas DataFrame with question1/2 in it
        te: test data frame
        bins: word shares are rounded to whole numbers after multiplying by bins.
        v_pos: number of virtual positives for smoothing (can be non-integer)
        vss: virtual sample size for smoothing (can be non-integer)
        
    Returns:
        submission in a Pandas Data Frame.
    '''
    tr['word_share'] = word_match_share(tr)
    tr['binned_share'] = (bins * tr.word_share).round()
    pos = tr.groupby('binned_share').is_duplicate.sum()
    cts = tr.binned_share.value_counts()
    te['word_share'] = word_match_share(te)
    te['binned_share'] = (bins * te.word_share).round()
    te_pos = te.binned_share.map(pos, na_action='ignore').fillna(0)
    te_cts = te.binned_share.map(cts, na_action='ignore').fillna(0)
    prob = (te_pos + vpos) / (te_cts + vss)
    odds = prob / (1 - prob)
    scaled_odds = SCALE * odds
    scaled_prob = scaled_odds / (1 + scaled_odds)
    sub = te[['test_id']].copy()
    sub['is_duplicate'] = scaled_prob
    return sub
sub = bin_model(tr, te)
sub.to_csv('no_ml_model.csv', index=False, float_format='%.6f')
sub.head(10)
