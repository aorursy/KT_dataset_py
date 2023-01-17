# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
root_path = '/kaggle/input/CORD-19-research-challenge/'
metadata_path = f'{root_path}/metadata.csv'
meta_df = pd.read_csv(metadata_path, dtype={
    'pubmed_id': str,
    'Microsoft Academic Paper ID': str, 
    'doi': str
})
Count = 1000#2168
meta_df = meta_df[meta_df["abstract"].notna()]
meta_df = meta_df[meta_df["cord_uid"].notna()]
meta_df = meta_df.head(Count)
meta_df["rank"] = range(1, len(meta_df)+1)
meta_df.set_index("cord_uid", inplace = True)
meta_df
import re
import string
import pickle

def clean_text_round1(text):
    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.''' 
    text = re.sub(r"$\d+\W+|\b\d+\b|\W+\d+$", " ", text)# remove numbers
    text = re.sub("([^\x00-\x7F])+"," ",text)#remove chinese and non ascii
    return text

round1 = lambda x: clean_text_round1(x)
# Let's take a look at the updated text
data_clean = pd.DataFrame(meta_df.abstract.apply(round1))
data_clean.head
# Apply a second round of cleaning
def clean_text_round2(text):
    '''Get rid of some additional punctuation and non-sensical text that was missed the first time around.'''
    text = re.sub('\n', '', text)
    return text

round2 = lambda x: clean_text_round2(x)
# Let's take a look at the updated text
data_clean = pd.DataFrame(data_clean.abstract.apply(round2))
data_clean
# Let's pickle it for later use
meta_df.to_pickle("corpus.pkl")
# We are going to create a document-term matrix using CountVectorizer, and exclude common English stop words
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(stop_words='english')
data_cv = cv.fit_transform(data_clean.abstract)
data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
data_dtm.index = data_clean.index
data_dtm.head()
# Let's pickle it for later use
data_dtm.to_pickle("dtm.pkl")
# Let's also pickle the cleaned data (before we put it in document-term matrix format) and the CountVectorizer object
data_clean.to_pickle('data_clean.pkl')
pickle.dump(cv, open("cv.pkl", "wb"))
# Read in the document-term matrix
data = pd.read_pickle('dtm.pkl')
data = data.transpose()
data.head()
#data
#data.iloc[:,2167]
#print(type(data.iloc[:,2167]))
#data.drop(data.columns[2167], axis=1, inplace=True)
# Find the top 30 words in each paper abstract
top_dict = {}
#print(data.iloc[:, [2017]])
#print(type(data))
#print(data.columns)
for c in data.columns:
#    print(data[c].head)
    #print(type(data[c]))
    if isinstance(data[c], pd.DataFrame)==False:
        top = data[c].sort_values(ascending=False).head(30)
        top_dict[c]= list(zip(top.index, top.values))

top_dict
# Print the top 15 words in each paper
for paper, top_words in top_dict.items():
    print(paper)
    print(', '.join([word for word, count in top_words[0:14]]))
    print('---')
# Look at the most common top words --> add them to the stop word list
from collections import Counter

# Let's first pull out the top 30 words for each paper
words = []
for paper in data.columns:
    if isinstance(data[paper], pd.DataFrame)==False:
        top = [word for (word, count) in top_dict[paper]]
        for t in top:
            words.append(t)
        
words
# If more than half of the papers have it as a top word, exclude it from the list
add_stop_words = [word for word, count in Counter(words).most_common(80) ]#if count > (Count//2)]
common_word_count = Counter(words).most_common()
#len(add_stop_words)
#add_stop_words
print(common_word_count)
import matplotlib.pyplot as plt
plt.rcdefaults()
fig, ax = plt.subplots(figsize=(18,50))
y_pos = [word for word, count in common_word_count]
count = [count for word, count in common_word_count]
plt.barh(y_pos[:80], count[:80], align='center')
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Count of words')
ax.set_title('Common words bar plot')
plt.show()
# Let's update our document-term matrix with the new list of stop words
from sklearn.feature_extraction import text 
from sklearn.feature_extraction.text import CountVectorizer

# Read in cleaned data
data_clean = pd.read_pickle('data_clean.pkl')

# Add new stop words
stop_words = text.ENGLISH_STOP_WORDS.union(add_stop_words)

# Recreate document-term matrix
cv = CountVectorizer(stop_words=stop_words)
data_cv = cv.fit_transform(data_clean.abstract)
data_stop = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
data_stop.index = data_clean.index

# Pickle it for later use
import pickle
pickle.dump(cv, open("cv_stop.pkl", "wb"))
data_stop.to_pickle("dtm_stop.pkl")
# Let's make some word clouds!
# Terminal / Anaconda Prompt: conda install -c conda-forge wordcloud
from wordcloud import WordCloud

wc = WordCloud(stopwords=stop_words, background_color="white", colormap="Dark2",
               max_font_size=150, random_state=42)
# Reset the output dimensions
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [16, 500]

cord_uid = list(meta_df.index.values)

# Create subplots for each paper
for index, paper in enumerate(data.columns):
    wc.generate(data_clean.abstract[paper])
    
    plt.subplot(250, 4, index+1)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(str(cord_uid[index]))
    
plt.show()
plt.tight_layout()
# Let's read in our document-term matrix
data = pd.read_pickle('dtm_stop.pkl')
data
# Import the necessary modules for LDA with gensim
# Terminal / Anaconda Navigator: conda install -c conda-forge gensim
from gensim import matutils, models
import scipy.sparse

# import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# One of the required inputs is a term-document matrix
tdm = data.transpose()
tdm.head()
# We're going to put the term-document matrix into a new gensim format, from df --> sparse matrix --> gensim corpus
sparse_counts = scipy.sparse.csr_matrix(tdm)
corpus = matutils.Sparse2Corpus(sparse_counts)
# Gensim also requires dictionary of the all terms and their respective location in the term-document matrix
cv = pickle.load(open("cv_stop.pkl", "rb"))
id2word = dict((v, k) for k, v in cv.vocabulary_.items())
# Now that we have the corpus (term-document matrix) and id2word (dictionary of location: term),
# we need to specify two other parameters as well - the number of topics and the number of passes
lda = models.LdaModel(corpus=corpus, id2word=id2word, num_topics=20, passes=10)
lda.print_topics()
# LDA for num_topics = 3
lda = models.LdaModel(corpus=corpus, id2word=id2word, num_topics=35, passes=20)
lda.print_topics()
# LDA for num_topics = 4
lda = models.LdaModel(corpus=corpus, id2word=id2word, num_topics=50, passes=50)
lda.print_topics()
# Let's create a function to pull out nouns from a string of text
from nltk import word_tokenize, pos_tag

def nouns(text):
    '''Given a string of text, tokenize the text and pull out only the nouns.'''
    is_noun = lambda pos: pos[:2] == 'NN'
    tokenized = word_tokenize(text)
    all_nouns = [word for (word, pos) in pos_tag(tokenized) if is_noun(pos)] 
    return ' '.join(all_nouns)
# Read in the cleaned data, before the CountVectorizer step
data_clean = pd.read_pickle('data_clean.pkl')
data_clean
# Apply the nouns function to the transcripts to filter only on nouns
data_nouns = pd.DataFrame(data_clean.abstract.apply(nouns))
data_nouns
# Create a new document-term matrix using only nouns
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer

# Re-add the additional stop words since we are recreating the document-term matrix
add_stop_words = ['like', 'im', 'know', 'just', 'dont', 'thats', 'right', 'people',
                  'youre', 'got', 'gonna', 'time', 'think', 'yeah', 'said']
stop_words = text.ENGLISH_STOP_WORDS.union(add_stop_words)

# Recreate a document-term matrix with only nouns
cvn = CountVectorizer(stop_words=stop_words)
data_cvn = cvn.fit_transform(data_nouns.abstract)
data_dtmn = pd.DataFrame(data_cvn.toarray(), columns=cvn.get_feature_names())
data_dtmn.index = data_nouns.index
data_dtmn
# Create the gensim corpus
corpusn = matutils.Sparse2Corpus(scipy.sparse.csr_matrix(data_dtmn.transpose()))

# Create the vocabulary dictionary
id2wordn = dict((v, k) for k, v in cvn.vocabulary_.items())
# Let's start with 2 topics
ldan = models.LdaModel(corpus=corpusn, num_topics=20, id2word=id2wordn, passes=10)
ldan.print_topics()
# Let's try topics = 3
ldan = models.LdaModel(corpus=corpusn, num_topics=35, id2word=id2wordn, passes=20)
ldan.print_topics()
# Let's try 4 topics
ldan = models.LdaModel(corpus=corpusn, num_topics=50, id2word=id2wordn, passes=50)
ldan.print_topics()
# Let's create a function to pull out nouns from a string of text
def nouns_adj(text):
    '''Given a string of text, tokenize the text and pull out only the nouns and adjectives.'''
    is_noun_adj = lambda pos: pos[:2] == 'NN' or pos[:2] == 'JJ'
    tokenized = word_tokenize(text)
    nouns_adj = [word for (word, pos) in pos_tag(tokenized) if is_noun_adj(pos)] 
    return ' '.join(nouns_adj)
# Apply the nouns function to the abstract to filter only on nouns
data_nouns_adj = pd.DataFrame(data_clean.abstract.apply(nouns_adj))
data_nouns_adj
# Create a new document-term matrix using only nouns and adjectives, also remove common words with max_df
cvna = CountVectorizer(stop_words=stop_words, max_df=.8)
data_cvna = cvna.fit_transform(data_nouns_adj.abstract)
data_dtmna = pd.DataFrame(data_cvna.toarray(), columns=cvna.get_feature_names())
data_dtmna.index = data_nouns_adj.index
data_dtmna
# Create the gensim corpus
corpusna = matutils.Sparse2Corpus(scipy.sparse.csr_matrix(data_dtmna.transpose()))

# Create the vocabulary dictionary
id2wordna = dict((v, k) for k, v in cvna.vocabulary_.items())
# Let's start with 2 topics
ldana = models.LdaModel(corpus=corpusna, num_topics=20, id2word=id2wordna, passes=10)
ldana.print_topics()
# Let's try 3 topics
ldana = models.LdaModel(corpus=corpusna, num_topics=35, id2word=id2wordna, passes=20)
ldana.print_topics()
# Let's try 4 topics
ldana = models.LdaModel(corpus=corpusna, num_topics=50, id2word=id2wordna, passes=50)
ldana.print_topics()
# Our final LDA model (for now)
ldana = models.LdaModel(corpus=corpusna, num_topics=50, id2word=id2wordna, passes=80)
ldana.print_topics()
# Let's take a look at which topics each transcript contains
corpus_transformed = ldana[corpusna]
t_lst = []
for tup_lst in corpus_transformed:
    t = list(map(lambda x: x[0], tup_lst))
    t_lst.append(t)
list(zip([a for a in t_lst], data_dtmna.index))
