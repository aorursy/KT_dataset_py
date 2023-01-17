# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



from collections import Counter

import string

import nltk

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/train.csv')

print('DataFrame Train:')

print(df_train.isnull().any())
toxic_comment = df_train[df_train['toxic'] == 1]['comment_text'].str.lower()

severe_toxic_comment = df_train[df_train['severe_toxic'] == 1]['comment_text'].str.lower()

obscene_comment = df_train[df_train['obscene'] == 1]['comment_text'].str.lower()

threat_comment = df_train[df_train['threat'] == 1]['comment_text'].str.lower()

insult_comment = df_train[df_train['insult'] == 1]['comment_text'].str.lower()

identity_hate_comment = df_train[df_train['identity_hate'] == 1]['comment_text'].str.lower()
toxic_comment = toxic_comment.values.tolist()

severe_toxic_comment = severe_toxic_comment.values.tolist()

obscene_comment = obscene_comment.values.tolist()

threat_comment = threat_comment.values.tolist()

insult_comment = insult_comment.values.tolist()

identity_hate_comment = identity_hate_comment.values.tolist()
toxic_comment_break = [nltk.tokenize.wordpunct_tokenize(text) for text in toxic_comment]

severe_toxic_comment_break = [nltk.tokenize.wordpunct_tokenize(text) for text in severe_toxic_comment]

obscene_comment_break = [nltk.tokenize.wordpunct_tokenize(text) for text in obscene_comment]

threat_comment_break = [nltk.tokenize.wordpunct_tokenize(text) for text in threat_comment]

insult_comment_break = [nltk.tokenize.wordpunct_tokenize(text) for text in insult_comment]

identity_hate_comment_break = [nltk.tokenize.wordpunct_tokenize(text) for text in identity_hate_comment]
stopwords = nltk.corpus.stopwords.words('english')
from itertools import groupby



def clear_multiple_char(comment):        

    ti = []

    for words in comment:

        t = [''.join(["".join(i) for i, _ in groupby(word)]) if len(word)>10 else word for word in words]

        ti.append(t)

    return ti
toxic_comment_break = clear_multiple_char(toxic_comment_break)

severe_toxic_comment_break = clear_multiple_char(severe_toxic_comment_break)

obscene_comment_break = clear_multiple_char(obscene_comment_break)

threat_comment_break = clear_multiple_char(threat_comment_break)

insult_comment_break = clear_multiple_char(insult_comment_break)

identity_hate_comment_break = clear_multiple_char(identity_hate_comment_break)
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

punctuation = string.punctuation # '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'

# Add numbers

punctuation += '0123456789'



def comment_raiz(comment):

    text = []

    for lista in comment:

        valids = [stemmer.stem(word) for word in lista if word not in stopwords and word not in punctuation 

                  and len(word)>2]

        valids_true = [''.join([char for char in word if char not in punctuation]) for word in valids if 

                       len(''.join([char for char in word if char not in punctuation]))>0]

        text.append(valids_true)

    return text
toxic_comment_clear = comment_raiz(toxic_comment_break)

severe_toxic_comment_clear = comment_raiz(severe_toxic_comment_break)

obscene_comment_clear = comment_raiz(obscene_comment_break)

threat_comment_clear = comment_raiz(threat_comment_break)

insult_comment_clear = comment_raiz(insult_comment_break)

identity_hate_comment_clear = comment_raiz(identity_hate_comment_break)
def counter(comment_clear):

    cnt = Counter()

    for words in comment_clear:

        for word in words:

            cnt[word] += 1

    return cnt
toxic_comment_cnt = counter(toxic_comment_clear)

severe_toxic_comment_cnt = counter(severe_toxic_comment_clear)

obscene_comment_cnt = counter(obscene_comment_clear)

threat_comment_cnt = counter(threat_comment_clear)

insult_comment_cnt = counter(insult_comment_clear)

identity_hate_comment_cnt = counter(identity_hate_comment_clear)
toxic_comment_cnt.most_common(10)
severe_toxic_comment_cnt.most_common(10)
obscene_comment_cnt.most_common(10)
threat_comment_cnt.most_common(10)
insult_comment_cnt.most_common(10)
identity_hate_comment_cnt.most_common(10)