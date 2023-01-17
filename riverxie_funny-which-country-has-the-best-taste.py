import pandas as pd
df=pd.read_csv('../input/winemag-data-130k-v2.csv',encoding='utf8')
df.dropna(axis=0, inplace=True, subset=['points'])

df.drop('Unnamed: 0',axis=1,inplace=True)
df.groupby('taster_name').count()
#see the how many reviews they contributed 

taster_count=df.groupby('taster_name').count()['points']

taster_count.describe(percentiles=[.01,.05, .5, .95,.99])
# remove the bottom 5% tasters

taster_count[taster_count>24.9].index
# remove the bottom 5% tasters

newdf = df[df['taster_name'].isin(taster_count[taster_count>24.9].index)]
newdf
%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns
import numpy as np

country_c = df.groupby('country')
country=country_c['points'].agg([np.sum, np.mean, np.std])
#rank the means points by country

country.sort_values(by='mean',ascending=False)[:20]
#plot top 20 countries

country.sort_values(by='mean',ascending=False)[:20]['mean'].plot.bar()
#rank the means points by region

region_1 = df.groupby('region_1')['points'].agg([np.sum, np.mean, np.std])
topRegion=region_1.sort_values(by='mean',ascending=False)[:20]

topRegion
for i in range(20):

    print(newdf[newdf['region_1']==topRegion.index[i]]['country'].unique(),topRegion.index[i])
import gensim.models.word2vec as w2v

import nltk

from nltk.corpus import stopwords

from nltk import FreqDist

import time,re
def sent_tokenizer(text):

    """

    Function to tokenize sentences

    """

    text = nltk.sent_tokenize(text)

    return text



def sentence_cleaner(text):

    """

    Function to lower case remove all websites, emails and non alphabetical characters

    """

    new_text = []

    for sentence in text:

        sentence = sentence.lower()

        sentence = re.sub("((\S+)?(http(s)?)(\S+))|((\S+)?(www)(\S+))|((\S+)?(\@)(\S+)?)", " ", sentence)

        sentence = re.sub("[^a-z ]", "", sentence)

        sentence = nltk.word_tokenize(sentence)

        sentence = [word for word in sentence if len(word)>1] # exclude 1 letter words

        new_text.append(sentence)

        #new_text = new_text+sentence

    return new_text

def apply_all(text):

    return sentence_cleaner(sent_tokenizer(text))
t1 = time.time()

newdf['sent_tokenized_desc'] = newdf['description'].apply(apply_all)

t2 = time.time()

print("time cost %.1f , records:%d"%((t2-t1)/60, len(newdf)))
newdf['sent_tokenized_desc'][0]
# create a list of all words using list comprehension

all_sentences = [word for item in list(newdf['sent_tokenized_desc']) for word in item]

all_words = [word for sent in all_sentences for word in sent]
all_words[:10]
fdist = FreqDist(all_words)

len(fdist) # number of unique words
# choose k and visually inspect the bottom 10 words of the top k

k = 10000

top_k_words = fdist.most_common(k)

top_k_words[-10:]
import multiprocessing
num_features = 300 # number of dimensions

# if any words appear less than min_word_count amount of times, disregard it

# recall we saw that the bottom 10 of the top 30,000 words appear only 7 times in the corpus, so lets choose 10 here

min_word_count = 5

num_workers = multiprocessing.cpu_count()

context_size = 7 # window size around target word to analyse

downsampling = 1e-3 # downsample frequent words

seed = 1 # seed for RNG
# setting up model with parameters above

desc2vec = w2v.Word2Vec(

    sg=1,

    seed=seed,

    workers=num_workers,

    size=num_features,

    min_count=min_word_count,

    window=context_size,

    sample=downsampling

)

desc2vec.build_vocab(all_sentences)
print("Word2Vec vocabulary length:", len(desc2vec.wv.vocab))
# train word2vec - this may take a minute...

desc2vec.train(all_sentences, total_examples=desc2vec.corpus_count, epochs=desc2vec.iter)
# dense 2D matrix of word vectors

all_word_vectors_matrix = desc2vec.wv.syn0
all_word_vectors_matrix.shape
all_word_vectors_matrix[desc2vec.wv.vocab['broom'].index]
desc2vec.wv.vocab['broom'].v
#concate sentences to description

tokenized_desc = []

for desc in list(newdf['sent_tokenized_desc']):

    text = []

    for sent in desc:

        text = text+sent

    tokenized_desc.append(text)
assert len(tokenized_desc)==len(newdf['sent_tokenized_desc'])
#vectorize the description

word_embedding_matrix = np.zeros((len(fdist), 300), dtype=np.float32)
vocab_to_int = {} 



value = 0

for word, count in fdist.items():

    vocab_to_int[word] = value

    value += 1

        

# Dictionary to convert integers to words

int_to_vocab = {}

for word, value in vocab_to_int.items():

    int_to_vocab[value] = word
embedding_dim = 300

nb_words = len(vocab_to_int)



# Create matrix with default values of zero

word_embedding_matrix = np.zeros((nb_words, embedding_dim), dtype=np.float32)

for word, i in vocab_to_int.items():

    if desc2vec.wv.vocab.get(word,0) != 0:

        word_embedding_matrix[i] = all_word_vectors_matrix[desc2vec.wv.vocab[word].index]

    else:

        # If word not in CN, create a random embedding for it

        new_embedding = np.array(np.random.uniform(-1.0, 1.0, embedding_dim))

        word_embedding_matrix[i] = new_embedding



# Check if value matches len(vocab_to_int)

print(len(word_embedding_matrix))