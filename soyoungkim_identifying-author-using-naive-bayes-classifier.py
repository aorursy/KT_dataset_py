# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

print ('There are {0} rows and {1} attributes.'.format(train.shape[0], train.shape[1]))

print ('There are {0} rows and {1} attributes.'.format(test.shape[0], test.shape[1]))
train.info()
if len(train['id'].unique()) == train.shape[0]:

    train.set_index(['id'], drop=False)
train_by_author = train.groupby('author')['text'].agg(lambda col: ''.join(col))
import os, re, math

from collections import Counter

from nltk.corpus import stopwords



def tokenizer(contents):

#     tokens = [] 

    parse = contents.replace('"','').replace(',','').replace('.','')

    tokens = re.sub('[^a-zA-Z0-9]', ' ', parse)

    tokens = tokens.lower().split()

    # remove remaining tokens that are not alphabetic

    tokens = [word for word in tokens if word.isalpha()]

    # filter out stop words

    stop_words = set(stopwords.words('english'))

    tokens = [w for w in tokens if not w in stop_words]

    # filter out short tokens

    tokens = [word for word in tokens if len(word) > 2]

    return tokens

   

EAP = tokenizer(train_by_author['EAP'])

MWS = tokenizer(train_by_author['MWS'])

HPL = tokenizer(train_by_author['HPL'])



vocab_EAP = Counter(list(EAP))

vocab_MWS = Counter(list(MWS))

vocab_HPL = Counter(list(HPL))



# print the size of the vocab

print ('EAP: {0}, MWS: {1}, HPL: {2} \n\n'.format(len(vocab_EAP), len(vocab_MWS), len(vocab_HPL)))
def naive_bayes_classifier(x):

    count = 0

    prob = 0

    for val in x:

        if count == 0:

#             prob = (math.log(0.33)) + (math.log(val))

            prob = 0.33 * val

        else:

#             prob = prob + (math.log(val))

            prob = prob * val



        count += 1

        

    return prob



def frequency_prob(x, couter, contents, smoothing):

    word_counts = []

    for word in x:

        word_counts.append((couter.setdefault(word, 0) + (math.pow(10, -200))) / (len(contents) + smoothing))

        

    return naive_bayes_classifier(word_counts)



sv_EAP = len(vocab_EAP) * (math.pow(10, -200))

sv_MWS  = len(vocab_MWS) * (math.pow(10, -200))

sv_HPL = len(vocab_HPL) * (math.pow(10, -200))

    

train['sentiment'] = train.apply(lambda row: tokenizer(row['text']), axis=1)

train['EAP'] = train.apply(lambda row: frequency_prob(row['sentiment'], vocab_EAP, EAP, sv_EAP), axis=1) 

train['MWS'] = train.apply(lambda row: frequency_prob(row['sentiment'], vocab_MWS, MWS, sv_MWS), axis=1) 

train['HPL'] = train.apply(lambda row: frequency_prob(row['sentiment'], vocab_HPL, HPL, sv_HPL), axis=1) 
def classifier(x, y, z):

    max_value = max(x, y, z)

    if x == max_value:

        return 'EAP'

    if y == max_value:

        return 'MWS'

    else:

        return 'HPL'



train['classifier'] = train.apply(lambda row: classifier(row['EAP'], row['MWS'], row['HPL']), axis=1)
# need to figure it out how to put into right probabilistic values (sum to 1)

train[['EAP', 'MWS', 'HPL']].head(20)
train['correctly_classified'] = np.where(train['author'] == train['classifier'], 1, 0)

ans = train[['correctly_classified']].values.sum()

print ('The accuracy of NB classifier model is {0}/{1}={2}.'.format(ans,

                                                                   train.shape[0],

                                                                   100*ans/train.shape[0]))