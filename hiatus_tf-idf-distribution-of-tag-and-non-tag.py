# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import nltk

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# read csv, print basic information. 

biology = pd.read_csv('../input/biology.csv')

print(biology.shape, type(biology))

print(biology.columns)
# extract and count tags from column 'tags'. 

tag_series = biology.get('tags')

tag_count = dict()

for _, tag_series in tag_series.iteritems():

    for tag in tag_series.split(' '):

        tag_count[tag] = tag_count.get(tag, 0) + 1



print(len(tag_count))
# generate tf-idf of title and content.

tac = biology.apply(lambda x: ' '.join([x['title'], x['content']]), 1)

tac = tac.apply(lambda x: x.lower())



# (optional)

# sk-learn's vectorizer will do lower() and clean markup. 

# tac = tac.apply(lambda x: x.lower())

# tac = tac.apply(lambda x: BeautifulSoup(x, 'html.parser').get_text())



vec_tfidf = TfidfVectorizer(min_df=4, dtype=np.float32)

x = vec_tfidf.fit_transform(tac.tolist())

x_col_names = vec_tfidf.get_feature_names()



word_count = pd.DataFrame({'word': x_col_names, 'count': x.sum(0).getA1()})
print(x.shape)
# split words to tags and non_tags. 

word_count['id'] = range(len(word_count))



word_is_tag = word_count[word_count.word.isin(tag_count.keys())]

word_not_tag = word_count[~word_count.word.isin(tag_count.keys())]

x_tags = x[:, word_is_tag['id'].tolist()]

x_not_tags = x[:, word_not_tag['id'].tolist()]
print(x_not_tags.shape, x_not_tags.mean())

print(x_tags.shape, x_tags.mean())
# plot distribution of tf-idf's which is positive.

hist_not_tag = plt.hist(x_not_tags[x_not_tags>0].getA1(), bins=200)

hist_tag = plt.hist(x_tags[x_tags>0].getA1(), bins=200, alpha=0.8)

plt.legend(['~tag[~tag>0]', 'tag[tag>0]'])
# plot distribution of tf-idf's mean of columns.

hist_not_tag = plt.hist(np.log2(x_not_tags.mean(0).getA1()), bins=200)

hist_tag = plt.hist(np.log2(x_tags.mean(0).getA1()), bins=200, alpha=0.8)

plt.legend(['log2(~tag.mean(0))', 'log2(tag.mean(0))'])