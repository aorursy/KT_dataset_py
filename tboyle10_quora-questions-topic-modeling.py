# import packages
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import spacy
import nltk
import re

from gensim import corpora, models, similarities
import pyLDAvis.gensim

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

np.random.seed(27)
# setting up default plotting parameters
%matplotlib inline

plt.rcParams['figure.figsize'] = [15.0, 7.0]
plt.rcParams.update({'font.size': 22,})

sns.set_palette('Set2')
sns.set_style('white')
sns.set_context('talk', font_scale=0.8)
# suppress warnings
import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head()
contractions = {
"ain't": "is not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"I'd": "I would",
"I'd've": "I would have",
"I'll": "I will",
"I'll've": "I will have",
"I'm": "I am",
"I've": "I have",
"i'd": "i would",
"i'd've": "i would have",
"i'll": "i will",
"i'll've": "i will have",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have"
}

c_re = re.compile('(%s)' % '|'.join(contractions.keys()))

def expandContractions(text, c_re=c_re):
    def replace(match):
        return contractions[match.group(0)]
    return c_re.sub(replace, text)
from gensim.parsing.preprocessing import preprocess_string
from gensim.parsing.preprocessing import strip_tags, strip_punctuation, strip_numeric, stem_text
from gensim.parsing.preprocessing import strip_multiple_whitespaces, strip_non_alphanum, remove_stopwords, strip_short

CUSTOM_FILTERS = [lambda x: x.lower(), #lowercase
                  strip_tags, # remove html tags
                  strip_punctuation, # replace punctuation with space
                  strip_multiple_whitespaces,# remove repeating whitespaces
                  strip_non_alphanum, # remove non-alphanumeric characters
                  strip_numeric, # remove numbers
                  remove_stopwords,# remove stopwords
                  strip_short, # remove words less than minsize=3 characters long
                  stem_text,
                 ]
def gensim_preprocess(docs):
    # clean text
    docs = [expandContractions(doc) for doc in docs]
    docs = [preprocess_string(text, CUSTOM_FILTERS) for text in docs]
    # create the bigram and trigram models
    bigram = models.Phrases(docs, min_count=1, threshold=1)
    trigram = models.Phrases(bigram[docs], min_count=1, threshold=1)  
    # phraser is faster
    bigram_mod = models.phrases.Phraser(bigram)
    trigram_mod = models.phrases.Phraser(trigram)
    # apply to docs
    docs = trigram_mod[bigram_mod[docs]]
    #docs = [' '.join(text) for text in docs]
    return docs

train_clean = gensim_preprocess(train.question_text)
train_clean[43]
# Create Dictionary from our ngram texts containing number of times token appears in training set
train_dictionary = corpora.Dictionary(train_clean)

# filter out extremes
train_dictionary.filter_extremes(no_below=0.1, # filter tokens appearing in <1% of documents
                                     no_above=0.7, # filter tokens appearing in >70% of documents
                                     keep_n=100000) # after above filters keep only the 100000 most frequent tokens

# For each document create dictionary with how many words and number of times the words appear
train_corpus = [train_dictionary.doc2bow(text) for text in train_clean]
# view human readable output
[[(train_dictionary[id], freq) for id, freq in cp] for cp in train_corpus[:1]]
# initialize tfidf model
tfidfi = models.TfidfModel(train_corpus)
# apply transformation to entire corpus
train_tfidf = tfidfi[train_corpus]
# https://radimrehurek.com/gensim/tut2.html#transformation-interface
# LDA on tfidf
%time train_lda = models.LdaMulticore(train_tfidf, num_topics=10, id2word=train_dictionary, passes=2, workers=6)
train_lda.show_topics()
# Visualize the topics
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(train_lda, train_corpus, train_dictionary)
vis
# using coherence score to find optimal number of topics
# ref: https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = models.LdaMulticore(corpus=corpus, num_topics=num_topics, id2word=dictionary, workers=6, passes=2)
        model_list.append(model)
        coherencemodel = models.CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

model_list, coherence_values = compute_coherence_values(dictionary=train_dictionary,
                                                        corpus=train_tfidf,
                                                        texts=train_clean,
                                                        start=2,
                                                        limit=262,
                                                        step=20)

coherence_values
# Show graph
limit=262; start=2; step=20;
x = range(start, limit, step)
sns.lineplot(x, coherence_values)
sns.despine(left=True, bottom=True)
plt.title('Training LDA Coherence Scores', fontsize=30)
plt.xlabel("Number of Topics")
plt.ylabel("Coherence Score")
plt.show()
# LDA on tfidf
%time train_lda = models.LdaMulticore(train_tfidf, num_topics=180, id2word=train_dictionary, passes=2, workers=6)
# Visualize the topics
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(train_lda, train_tfidf, train_dictionary)
vis

