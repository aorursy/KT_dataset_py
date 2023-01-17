import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

from textblob import TextBlob

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Create pools

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# months = [str(i).zfill(2) for i in range(1, 13)]

days = [str(i).zfill(2) for i in range(1, 29)]

years = [str(i) for i in range(1992, 2016)]
# Randomly generate from pools

rand_months = np.random.choice(months, size=10000)

rand_days = np.random.choice(days, size=10000)

rand_years = np.random.choice(years, size=10000)
# Place into df

%time rand_dates = pd.DataFrame({'yr':rand_years, 'mo':rand_months, 'da': rand_days})
rand_dates.head()
%timeit rand_dates['mo_da']  = rand_dates['mo'].str.cat(rand_dates['da'])
%timeit rand_dates['mo_da2']  = rand_dates['mo'] + rand_dates['da']
%timeit rand_dates['mo'].str.slice(0,1) + rand_dates['da'].str.slice(0,1)
%timeit rand_dates['mo'].str[0] + rand_dates['da'].str[0]
%timeit rand_dates['mo'].str.get(0) + rand_dates['da'].str.get(0)
rand_dates.loc[23, 'mo'] = ''
%timeit rand_dates['mo'].str.slice(0,1) + rand_dates['da'].str.slice(0,1)
rand_dates.loc[23, 'mo'] = np.NaN
%timeit rand_dates['mo'].str.slice(0,1) + rand_dates['da'].str.slice(0,1)
%timeit rand_dates['mo'] + rand_dates['da']
train = pd.read_csv('../input/train.csv', nrows=10000)

train2 = train.copy()
%timeit d = train['question_text'].str.slice(0,1) # same trend as earlier
%timeit a = train['question_text'].str[0]
%timeit b = train['question_text'].str.count('e')
%timeit c = train['question_text'].str.capitalize()
%%timeit

train['first'] = train['question_text'].str[0]

train['count_e'] = train['question_text'].str.count('e')

train['cap'] = train['question_text'].str.capitalize()

# Just the individual values added together
def extract_text_features(x):

    return x[0], x.count('e'), x.capitalize()
%timeit train['first'], train['count_e'], train['cap'] = zip(*train['question_text'].apply(extract_text_features))
%%timeit

a,b,c = [], [], []

for s in train['question_text']:

    a.append(s[0]), b.append(s.count('e')), c.append(s.capitalize())

train['first'] = a

train['count_e'] = b

train['cap'] = c

# assigning to new column takes about the same time in either method
%timeit x = train['question_text'].str.len()
%timeit b = train['question_text'].apply(lambda x:len(x))
# bonus - getting memory of your array

train['question_text'].values.nbytes
%%timeit 

train2['num_chars'] = train2['question_text'].str.len()

train2['is_titlecase'] = train2['question_text'].str.istitle().astype('int')

train2['has_*'] = train2['question_text'].str.contains(r'[A-Za-z]\*.|.\*[A-Za-z]', regex=True).astype('int')

def srs_funcs(srs):

    a = len(srs)

    b = int(srs.istitle())

    c = int(bool(re.search(r'[A-Za-z]\*.|.\*[A-Za-z]', srs)))

    return a, b, c

# would have expected this to be faster than creating three new columns individually but maybe the type conversion calls slowed it down
%timeit  train2['num_chars'] , train2['is_titlecase'], train2['has_*'] = zip(*train2['question_text'].apply(srs_funcs))
def srs_funcs2(srs):

    a = len(srs)

    b = int(srs.istitle())

    c = int(bool(re.search(r'[A-Za-z]\*.|.\*[A-Za-z]', srs)))

    return pd.Series([a, b, c])
%timeit  train2[['num_chars','is_titlecase','has_*']] = train2['question_text'].apply(srs_funcs2)

# calling pd.series each time through loop kills performance
def textblob_methods(blob):

    '''Access Textblob methods and returns as tuple

    '''

    # convert to python list of tokens

    return blob.polarity, blob.subjectivity, int(blob.ends_with('?'))
train3 = pd.read_csv('../input/train.csv', nrows=10000)

train3.head()
# Convert  - any ways to make this faster? 

%timeit train3['blobs'] = train3['question_text'].map(lambda x: TextBlob(x))
%timeit zsamp = train3.loc[5006,'blobs']
%timeit zsamp = train3.loc[5006]['blobs']
zsamp = train3.loc[5006]['blobs']
%timeit textblob_methods(zsamp)
%timeit  train3['polarity'], train3['subjectivity'], train3['ends_with_?'] = zip(*train3['blobs'].map(textblob_methods))
%%timeit

a, b, c = [], [], []

for s in train3['blobs']:

    a.append(s.polarity), b.append(s.subjectivity), c.append(int(s[-1] in '?'))

train3['polarity'], train3['subjectivity'], train3['ends_with_?'] = a, b, c
%%timeit

# Doing it separately - takes longer

train3['polarity'] = train3['blobs'].apply(lambda x: x.polarity)

train3['subjectivity'] = train3['blobs'].apply(lambda x: x.subjectivity)

train3['ends_with_?'] = train3['blobs'].apply(lambda x: x.endswith('?'))
def textblob_methods2(blob):

    '''Access Textblob methods and returns as tuple

    '''

    # convert to python list of tokens

    return blob.polarity, blob.subjectivity
%timeit  train3['polarity'], train3['subjectivity'] = zip(*train3['blobs'].map(textblob_methods2))
%%timeit

a, b = [], []

for s in train3['blobs']:

    a.append(s.polarity), b.append(s.subjectivity)

train3['polarity'], train3['subjectivity'] = a, b