# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re



def extract_csv_data(file, cols_to_clean = [], exclude = [[]]):

    data = pd.read_csv(file)



    for i, col in enumerate(cols_to_clean):

        exclude_pattern = re.compile('|'.join(exclude[i]))

        data = data[data[col].str.contains(exclude_pattern) == False]



    return data



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



emails = extract_csv_data(

    '../input/emails.csv',

    ['file'],

    [['notes_inbox', 'discussion_threads']]

)



def remove_duplicates(data):

    processed = set()

    result = []

    pattern = re.compile('X-FileName: .*')

    pattern2 = re.compile('X-FileName: .*?  ')



    for doc in data:

        doc = doc.replace('\n', ' ')

        doc = doc.replace(' .*?nsf', '')

        match = pattern.search(doc).group(0)

        match = re.sub(pattern2, '', match)



        if match not in processed:

            processed.add(match)

            result.append(match)



    return result



email_bodies = emails.message.as_matrix()

unique_emails = remove_duplicates(email_bodies)



print('There are a total of {} non-duplicate emails\n'.format(len(unique_emails)))

print('Sample email, unstructured content:\n\n', unique_emails[1000])
from nltk.corpus import stopwords

from nltk.stem.wordnet import WordNetLemmatizer

import string



def clean(doc):

    words_to_exclude = set(stopwords.words('english'))

    exclude = set(string.punctuation)

    lemma = WordNetLemmatizer()



    word_free = " ".join([i for i in doc.lower().split() if i not in words_to_exclude])

    punc_free = ''.join(ch for ch in word_free if ch not in exclude)

    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())



    return normalized



def clean_data(data):

    return [clean(doc).split(' ') for doc in data]



training_set = clean_data(unique_emails[0:200000])

testing_set = clean_data(unique_emails[200000:])



print(training_set[1000])
import gensim



dictionary = gensim.corpora.Dictionary(training_set)

dictionary.filter_extremes(no_below=20, no_above=0.1)



print(dictionary)
matrix = [dictionary.doc2bow(doc) for doc in training_set]



tfidf_model = gensim.models.TfidfModel(matrix, id2word=dictionary)

lsi_model = gensim.models.LsiModel(tfidf_model[matrix], id2word=dictionary, num_topics=100)



topics = lsi_model.print_topics(num_topics=100, num_words=10)

for topic in topics:

    print(topic)