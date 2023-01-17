import numpy as np

import pandas as pd

import re



df = pd.read_csv("../input/twitter-sentiment-analysis-hatred-speech/train.csv")



# helper function to remove twitter handles

def remove_pattern(input_text, pattern):

    r = re.findall(pattern, input_text) # retuns a list with substrings with 'pattern'

    for i in r:

        input_text = re.sub(i, "", input_text) # remove pattern 

    return input_text



# to lower

df['to_lower'] = df['tweet'].apply(lambda x: x.lower())

# find pattern for twitter handles using regex

# @user

df['handle_removed'] = np.vectorize(remove_pattern)(df['to_lower'], "@[\w]*")

# 1. convert pandas series to string

# 2. call replace method on string

# 3. Use regex to replace everything except [a-z] and [A-Z] with space (" ")

# 4. use "[^a-zA-Z#]" to retain hash-symbol (not doing it here)

df['puncs_removed'] = df['handle_removed'].str.replace("[^a-zA-Z]", " ")



df.head(3)
# to numpy array

df['puncs_removed'].values[:3]
len(df['puncs_removed'])
from sklearn.feature_extraction.text import CountVectorizer



# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html

bow = CountVectorizer(

            stop_words     = 'english',

            binary         = False, # True -> Binary BoW,

            ngram_range    = (1,1), # (1, 1) -> only unigrams, (1, 2) -> unigrams and bigrams, and (2, 2) -> only bigrams

            #vocabulary    = Mapping / iterable (custom vocabulary)

        )
# use `fit_transform` with training data (seeing first time => generates vocabulary)

# use `transform` with test data (not seeing first time => needs vocabulary - already genearated by `fit_transform`)

features = bow.fit_transform(df['puncs_removed'].values)



print(features.shape)

print(type(features))
print(dir(bow))
print("vocabulary size is: ", len(bow.vocabulary_))



# method 1:

# out of 37255 

print("method 1:")

for vocab_word, index in bow.vocabulary_.items():

    if index == 0   : print("feature 0 repesents word: ", vocab_word)

    if index == 100 : print("feature 100 repesents word: ", vocab_word)

        

# method 2: 

print("method 2:")

print(f"feature 0 repesents word: {bow.get_feature_names()[0]}")

print(f"feature 100 repesents word: {bow.get_feature_names()[100]}")
# `transform` instead of `fit_transform` for test-data

bow.transform(['hi']) # (1x37255)
from nltk.stem import PorterStemmer, SnowballStemmer

from nltk.corpus import stopwords



''' 

# Lemmatisation:

# base word of stem might not be an actual word whereas, lemma is an actual language word



>>> from nltk.stem import WordNetLemmatizer

>>> wnl = WordNetLemmatizer()

>>> print(wnl.lemmatize('dogs'))

dog

>>> print(wnl.lemmatize('churches'))

church

>>> print(wnl.lemmatize('aardwolves'))

aardwolf



USAGE: instead of stemming pd.series below, use

`.apply(lambda x: [wnl.lemmatize(i) for i in x])`

'''





stopwords = set( stopwords.words('english') )



# SnowballStemmer stemmer better

#stemmer = PorterStemmer('english')

stemmer = SnowballStemmer('english')



# Tokenize before stemming. 

# Tokenize: Split into particular words i.e into list

tokenized_tweet = df['puncs_removed'].apply(lambda x: x.split())



# Stopword removal (in-place)

tokenized_tweet = tokenized_tweet.apply(lambda tokens: [i if i not in stopwords else '' for i in tokens])



# Stemming (can be replaced w/ Lemmatisation)

# Iterate over every word in each list 

# So that `having` and `have` both can be converted into `have`

stemmed_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x])



# convert list of words into a line

for i in range(len(stemmed_tweet)):

    stemmed_tweet[i] = ' '.join(stemmed_tweet[i])

df["processed"] = stemmed_tweet



# display

df[['puncs_removed', 'processed']].head(3)
from sklearn.feature_extraction.text import TfidfVectorizer



# use exactly same as BoW

# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

tfidf = TfidfVectorizer(

            stop_words   = 'english',

            ngram_range  = (1,2) # uni as well as bi

        )
# use `fit_transform` with training data (seeing first time => generates vocabulary)

# use `transform` with test data (not seeing first time => needs vocabulary - already genearated by `fit_transform`)

tfidf_features = tfidf.fit_transform(df['processed'].values)



print(tfidf_features.shape)

print(type(tfidf_features))
def get_topn_tfidfs_of_a_sample(tfidf_features, sample_idx, n=25):

    """

    tfidf_features  : np.ndarray of dims (num_samples, num_tfidf_feats)

    sample_idx      : row_idx

    """

    tfidfs_of_a_row = tfidf_features[sample_idx].toarray().flatten() # (1x31194) -> (31194,)

    desc_idxs = np.argsort(tfidfs_of_a_row)[::-1][:n]

    

    top_n_tfidfs = tfidfs_of_a_row[desc_idxs]

    top_n_tfidfs_featwords = np.array(tfidf.get_feature_names())[desc_idxs]

    

    return top_n_tfidfs_featwords, top_n_tfidfs
# Analyze top-n TF-IDFs of a >>data-sample 0 and 1<<

# inspiration: http://buhrmann.github.io/tfidf-analysis

bar_xs_0, bar_ys_0 = get_topn_tfidfs_of_a_sample(tfidf_features, 0, n=25)

bar_xs_1, bar_ys_1 = get_topn_tfidfs_of_a_sample(tfidf_features, 1, n=25)
import matplotlib.pyplot as plt

import seaborn as sns



fig, axarr = plt.subplots(1, 2)

fig.set_size_inches(12,5)



# sample at idx 0

sns.barplot(bar_ys_0, bar_xs_0, ax=axarr[0])

axarr[0].set_title('For sample at idx 0')

axarr[0].grid()



# sample at idx 1

sns.barplot(bar_ys_1, bar_xs_1, ax=axarr[1])

axarr[1].set_title('For sample at idx 1')

axarr[1].grid()



fig.tight_layout(pad=3.0)

plt.show()
! curl -O "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
from gensim.models import Word2Vec, KeyedVectors

pretrained_model = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin.gz', binary=True)
# word -> 300 dim vec (pretrained)

pretrained_model['test'].shape
# similarity using norm of distance vector 

# output is normalized (between 0,1)

pretrained_model.similarity('king', 'queen')
pretrained_model.most_similar('king')
sentences = df['processed'].values # ndarray of stemmed



list_of_list_of_list_of_words = []

for sentence in sentences:

    list_of_list_of_list_of_words.append(

        sentence.split()

    )

    

list_of_list_of_list_of_words[:2]
# train custom model

custom_model = Word2Vec(

                list_of_list_of_list_of_words,

                min_count     = 1,

                size          = 300,

                workers       = 4

                )
custom_model.wv['lyft'].shape # custom vocabulary
custom_model.wv.most_similar('lyft')
list_of_sentences = df['processed'].values # ndarray of stemmed sentences

list_of_sentences[0]
from tqdm import tqdm



avg_w2v_sentences = []

for sentence in tqdm(list_of_sentences):

    w2vs = []

    for word in sentence.split():

        w2vs.append(custom_model.wv[word])

        

    w2vs = np.array(w2vs)

    avg_w2v_sentence_vec = np.sum(w2vs, axis=0) / len(sentence)

    avg_w2v_sentences.append(avg_w2v_sentence_vec)
avg_w2v_sentences = avg_w2v_sentences



print(avg_w2v_sentences[0].shape)

print(len(avg_w2v_sentences))