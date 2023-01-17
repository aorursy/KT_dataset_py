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
data = pd.read_csv("../input/abcnews-date-text.csv", error_bad_lines=False)
data.head()
text = data[['headline_text']]
text.head()
text['index'] = text.index
text.head()
documents = text
documents.head()
print("Total length of the documents: {}".format(len(documents)))
# importing the gensim and nltk libraries



import gensim

from gensim.utils import simple_preprocess

from gensim.parsing.preprocessing import STOPWORDS

from nltk.stem import WordNetLemmatizer, SnowballStemmer

from nltk.stem.porter import *



import nltk

np.random.seed(42)
def preprocessing(sentence):

    stemmer = SnowballStemmer('english')

    return stemmer.stem(WordNetLemmatizer().lemmatize(sentence, pos='v'))



def preprocess(sentence):

    result = []

    

    for token in gensim.utils.simple_preprocess(sentence):

        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:

            result.append(preprocessing(token))

            

    return result
sample = documents[documents['index'] == 4310].values[0][0]



print("Sample document is selected for pre-processing: {}".format(sample))
words = []



for word in sample.split(' '):

    words.append(word)

    

print("Words found after splitting the sample document: {}".format(words))
print("Tokenized and lemmatized document: {}".format(preprocess(sample)))
# pre-processing all the documents



preprocessed_documents = documents['headline_text'].map(preprocess)
preprocessed_documents[:10]
# creating a dictionary from the above processed documents



dictionary = gensim.corpora.Dictionary(preprocessed_documents)
count = 0



for key, value in dictionary.iteritems():

    print("Key: {} and Value: {}".format(key, value))

    count += 1

    

    if count > 10:

        break
# filter out extreme tokens in the document



dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
bag_of_words = [dictionary.doc2bow(document=document) for document in preprocessed_documents]
bag_of_words[4310]
## preview of bag of words of our sample preprocessed document



sample_bag_of_words = bag_of_words[4310]



for i in range(len(sample_bag_of_words)):

    print("Word: {} (\"{}\") appears: {} times.".format(sample_bag_of_words[i][0], dictionary[sample_bag_of_words[i][0]], sample_bag_of_words[i][1]))
from gensim import corpora, models



tfidf = models.TfidfModel(bag_of_words)

corpus_tfidf = tfidf[bag_of_words]



from pprint import pprint



for document in corpus_tfidf:

    pprint(document)

    break
# training our model using gensim LdaMulticore



model = gensim.models.LdaMulticore(bag_of_words, num_topics=10, id2word=dictionary, passes=2, workers=2)
for index, topic in model.print_topics(-1):

    print("Topic: {} \n Words: {}".format(index, topic))
tfidf_model = gensim.models.LdaMulticore(corpus_tfidf, num_topics=10, id2word=dictionary, passes=2, workers=4)



for index, topic in tfidf_model.print_topics(-1):

    print("Topic: {}, Words: {}".format(index, topic))
preprocessed_documents[4310]
for index, score in sorted(tfidf_model[bag_of_words[4310]], key=lambda tup: -1 * tup[1]):

    print("\nScore: {} \t \nTopic: {}".format(score, tfidf_model.print_topics(index, 10)))
test_document = "How a Pentgon deal became an identity crisis for Google"



bag_of_words_vector = dictionary.doc2bow(preprocess(test_document))
for index, score in sorted(tfidf_model[bag_of_words_vector], key=lambda tup: -1 * tup[1]):

    print("Score: {} \t Topic: {}\n".format(score, tfidf_model.print_topics(index, 5)))